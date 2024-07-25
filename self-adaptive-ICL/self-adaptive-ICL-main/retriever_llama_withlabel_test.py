import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ['WANDB_PROJECT'] = 'ICL'
os.environ['WANDB_ENTITY'] = 'zixiaozhu'
os.environ['WANDB_API_KEY'] = '08231f47d978105125589d4ff268a928417cd933'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HYDRA_FULL_ERROR'] = '1'
import glob
import json
import warnings
import logging
from typing import Optional, Dict, List
import hydra
import tqdm
from accelerate import Accelerator

from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from inferencer import Inferencer
from src.datasets.instructions import *
from src.models.model import evaluate, get_score, generate
from src.utils.calculate import transform, get_permutations

import random
import numpy as np

logger = logging.getLogger(__name__)

class Retriever(Inferencer):

    def __init__(self, cfg, accelerator) -> None:
        super(Retriever, self).__init__(cfg, accelerator)
        self.window = cfg.window
        self.input_file = cfg.dataset_reader.dataset_path
        self.output_file = cfg.output_file
        self.method = cfg.method
        self.num_ice = cfg.num_ice
        self.instruction_template = cfg.instruction_template
        self.force_topk = cfg.force_topk
        self.span = cfg.span
        self.n_tokens = cfg.n_tokens
        self.printonce = 0
        self.all_permutation = cfg.all_permutation
        self.calibrate = cfg.calibrate
        self.sort = cfg.sort
        self.use_rand_pool = cfg.use_rand_pool
        self.rand_pool = None
        self.prior_no = cfg.prior_no

        instructions = get_template(self.task_name, self.instruction_template)
        self.labels = [y for y in instructions.keys()]
        # self.example_instruction={}
        # self.prompting_instruction={}
        # instructions=None
        if instructions is not None:
            self.labels = [y for y in instructions.keys()]  # [int]    QA:[0]
            self.example_instruction = {label: instructions[label]['example_instruction'] for label in self.labels}
            self.prompting_instruction = {label: instructions[label]['prompting_instruction'] for label in self.labels}
        else:

            self.example_instruction = {label: '' for label in self.labels}
            self.prompting_instruction = {label: '' for label in self.labels}

        self.task_type = get_task_type(self.task_name)
        if self.task_type == "QA":
            self.tokenizer.padding_side = "left"


    def forward(self):

        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        tmpfile = f"{self.output_file}tmp_{self.accelerator.device}.bin"
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:

            for i, entry in enumerate(dataloader):
                metadata = entry.pop("metadata")
                if self.printonce > 0:
                    self.printonce -= 1
                metadata = transform(metadata)
                if self.printonce > 0:
                    self.printonce -= 1

                ctxs = [self.retrieve(pool=m["examples"], num=self.num_ice, query=m,method=self.method)[0] for m in metadata]
                ctxs_score = [self.retrieve(pool=m["examples"], num=self.num_ice, query=m,method=self.method)[1] for m in metadata]

                for mdata, selected_idx in zip(metadata, ctxs):
                    mdata.pop("examples")
                    mdata['selected_idxs'] = selected_idx
                    mdata['selected_idxs_top1score'] = ctxs_score[0]
                    for key in mdata.keys():
                        if torch.is_tensor(mdata[key]):
                            mdata[key] = mdata[key].tolist()
                    buffer.write(mdata)

    def retrieve(self, pool: List[Dict], num: int, query: Optional[Dict] = None, method: Optional[str] = None):
        selected_idxs,selected_score = self.instance_level_lm_score(pool=pool, num=num, query=query, method=method)
        return selected_idxs,selected_score

    def instance_level_lm_score(self, pool: List[Dict], num: int, query, method: str = 'mdl') \
            -> List:
        window = self.window  # number of candidates

        if self.use_rand_pool:
            if self.rand_pool is None:
                self.rand_pool = [np.random.choice(list(range(len(pool))), size=num, replace=False).tolist() for _ in
                                  range(window)]
            all_candidate_idx = self.rand_pool

        elif self.all_permutation:
            all_candidate_idx = get_permutations(num)
        elif self.sort:
            all_candidate_idx = [sorted(np.random.choice(list(range(len(pool))), size=num, replace=False).tolist())
                                 for _ in range(window)]
        else:
            all_candidate_idx = [np.random.choice(list(range(len(pool))), size=num, replace=False).tolist() for _ in
                                 range(window)]
        if self.force_topk:
            new = [i for i in range(num)]
            all_candidate_idx.pop(0)
            all_candidate_idx.append(new)
        if window == 1:
            return all_candidate_idx[0]

        in_context_examples = [[pool[i] for i in candidates_idx] for candidates_idx in all_candidate_idx]

        if self.task_type == "QA":
            batch = [build_instruction(x=query['X'], c=query['C'], e=e, y_text="",
                                       instruction=self.prompting_instruction[0],
                                       tokenizer=self.tokenizer,
                                       e_instruction=self.example_instruction, need_span_ids=self.span,
                                       max_len=self.n_tokens)[0]
                     for e in in_context_examples]
            generated = generate(self, batch, span=self.span)
        choice = self.task_type == "CHOICE"
        choice_sep = get_choice_sep(self.task_name)

        batch_labels = [[build_instruction(x=query['X'], c=query['C'], e=e,
                                           y_text=None if self.task_type != "QA" else generated[i],
                                           instruction=self.prompting_instruction[label],
                                           tokenizer=self.tokenizer,
                                           e_instruction=self.example_instruction, need_span_ids=self.span,
                                           max_len=self.n_tokens,choice=choice,choice_sep=choice_sep)[0]
                         for i, e in enumerate(in_context_examples)]
                        for label in self.labels]  # label:int
        if self.calibrate and self.task_type != 'QA':
            batch_labels_prior = [[build_instruction(x=query['X'], c=query['C'], e=e,
                                                     y_text=None,
                                                     instruction=self.prompting_instruction[label],
                                                     tokenizer=self.tokenizer,
                                                     e_instruction=self.example_instruction, need_span_ids=self.span,
                                                     max_len=self.n_tokens, prior=True, prior_no=self.prior_no)[0]
                                   for i, e in enumerate(in_context_examples)]
                                  for label in self.labels]  # label:int
            prior_loss_list = []
            for batch in batch_labels_prior:
                with torch.no_grad():
                    prior_ce_loss, prior_lens = evaluate(self, batch, span=self.span)
                    avg_prior_loss = (prior_ce_loss / prior_lens).tolist()
                    prior_loss_list.append(avg_prior_loss)
            scores = get_score(self, batch_labels, method=method, span=self.span, prior_loss_list=prior_loss_list)
        else:
            scores = get_score(self, batch_labels, method=method, span=self.span)

        selected_idxs = all_candidate_idx[scores.argmax()]
        return selected_idxs,max(scores)  # list int

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            print(path)
            with BufferedJsonReader(path) as f:
                data.extend(f.read())
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)


        outputfile1 = self.input_file
        with open(outputfile1, 'r') as load_f:
            load_data = json.load(load_f)
        data.sort(key=lambda x: int(x['id']))

        ctxs_list = [np.array(load_data[i]["ctxs_candidates"]).reshape(-1)[data[i]["selected_idxs"]]
                     for i in range(len(load_data))]
        score_list = [data[i]["selected_idxs_top1score"]for i in range(len(load_data))]
        count=0
        with open(self.output_file, "w") as f:
            for d, ctxs in zip(load_data, ctxs_list):
                d["ctxs"] = ctxs.tolist()
                d['mdl_score']=score_list[count]
                count+=1
            json.dump(load_data, f)


        return data


@hydra.main(config_path="configs", config_name="retriever")
def main(cfg):
    for task_name in [
        # 'cr',
        # 'aman',
        # 'trec',
        # 'sst2',
        # 'isear',
        # 'imdb',
        'ag_news'
    ]:
        if task_name=='isear':
            label_dict = {
                'fear': 0,
                          'sadness': 1,
                'disgust': 2,
                'anger': 3,
                'joy': 4,
                'guilt': 5,
                'shame': 6
            }
        elif task_name=='aman':
            label_dict = {
                'fear': 0,
                'sadness': 1,
                'disgust': 2,
                'anger': 3,
                'joy': 4,
                'surprise': 5,
                'others': 6
            }
        elif task_name == 'cr' or task_name == 'sst2' or task_name == 'imdb':
            label_dict = {
                'negative': 0,
                'positive': 1,

            }
        elif task_name == 'trec':
            label_dict = {
                'abbreviation': 0,
                'entity': 1,
                'description': 2,
                'human': 3,
                'location': 4,
                'number': 5
            }
        elif task_name == 'ag_news':
            label_dict = {
                'Worlds': 0,
                'Sports': 1,
                'Business': 2,
                'Technology': 3,

            }
        for key in label_dict:
            print(key)

            root = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/'
            num_ice = 8
            num_candidates = 30
            prerank_method = 'topk'
            score_method = 'mdl'

            # model_name = 'gpt2-xl'
            model_name = 'meta-llama/Llama-2-7b-chat-hf'
            n_tokens = 700
            inf_batch_size = 12
            instruction_template = 1
            span = True
            window = 10
            dataset_split = "test"
            rand_seed = 1
            port = 12715
            emb_field = "X"
            n_gpu = 1
            # run_dir = os.path.join(root, f'output/{task_name}/{model_name}/{rand_seed}/{dataset_split}/test/')
            run_dir = os.path.join(root, f'output/{task_name}/test/1000')
            retrieve_file = os.path.join(run_dir, f'retrieved{key}.json')
            retrieve_file2 = os.path.join(run_dir, f'retrieved2{key}.json')
            pred_file = os.path.join(run_dir, f'pred4{key}.json')

            os.makedirs(run_dir, exist_ok=True)

            cfg.output_file = retrieve_file2
            cfg.window = window
            cfg.num_ice = num_ice
            cfg.rand_seed = rand_seed
            cfg.instruction_template = instruction_template
            cfg.dataset_reader.task_name = task_name
            cfg.span = span
            cfg.dataset_reader.dataset_path = retrieve_file
            cfg.batch_size = inf_batch_size
            cfg.method = score_method
            # cfg.dataset_reader.index_data_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/train_{task_name}_{key}.csv'
            cfg.dataset_reader.index_data_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{task_name}_1000_{key}.csv'
            # cfg.dataset_reader.dataset_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/valid/valid_{task_name}_prompt.csv'
            cfg.task_name = task_name
            logger.info(cfg)
            if not cfg.overwrite:
                if os.path.exists(cfg.output_file):
                    logger.info(f'{cfg.output_file} already exists,skip')
                    return
            random.seed(cfg.rand_seed)
            np.random.seed(cfg.rand_seed)
            accelerator = Accelerator()
            retriever = Retriever(cfg, accelerator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                retriever.forward()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    retriever.write_results()


if __name__ == "__main__":
    main()
