import pickle
import warnings
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
from ..lm_apis.lm_api_base import LMForwardAPI
from ..utils.data_wrapper import wrap_dataset, tokenize_dataset
from ..utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from ..utils.random_utils import set_seed
from ..utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class ,dict_to
from transformers import Trainer, TrainingArguments
from ..utils.load_local import get_model_layer_num
from ..util_classes.arg_classes import ReweightingArgs
from ..utils.prepare_model_and_tokenizer import load_model_and_tokenizer, get_label_id_dict_for_args,get_label_id_dict_for_args_all
from ..util_classes.predictor_classes import Predictor
from .attentioner_for_train import GPT2AttentionerManager,Llama2AttentionerManager
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch

def get_acc(y,labels):
    scores = y.predictions[0]
    predict= np.argmax(scores, axis=1)
    false_list=[]
    for l in range(len(labels)):
        if int(labels[l])!=int(predict[l]):
            false_list.append(l)
    acc = accuracy_score(labels, np.argmax(scores, axis=1))
    # r = recall_score(labels, np.argmax(scores, axis=1), average='weighted')
    # p = precision_score(labels, np.argmax(scores, axis=1), average='weighted')
    # f1 = f1_score(labels, np.argmax(scores, axis=1), average='weighted')
    return acc,false_list



def train(train_file,test_file,save_file,multi_token,args: ReweightingArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from=='mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)
    elif args.sample_from == 'mydataset_define_demo':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        # dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        # dataset_test = Dataset.from_pandas(dftest)


    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)
    n_class_tokens=0
    for k in args.label_id_dict_all:
        n_class_tokens+=len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token,model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_device_eval_batch_size=args.batch_size,
                                      per_device_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)
    def prepare_analysis_dataset(seed):
        demonstration, train_samples = sample_two_set_with_shot_per_class(dataset['train'],
                                                                          args.demonstration_shot,
                                                                          args.train_num_per_class,
                                                                          seed,
                                                                          label_name='label',
                                                                          a_total_shot=args.demonstration_total_shot)
        if args.sample_from == 'test':
            if len(dataset['test']) < args.actual_sample_size:
                args.actual_sample_size = len(dataset['test'])
                warnings.warn(
                    f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                    f"actual_sample_size is {args.actual_sample_size}")
            # test_sample = dataset['test'].shuffle(seed=seed).select(range(args.actual_sample_size))
            test_sample = dataset['test']
            dataset_statistics(test_sample['label'])
            analysis_dataset = wrap_dataset(test_sample, demonstration, args.label_dict,
                                            args.task_name)
            analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

            train_dataset = wrap_dataset(train_samples, demonstration, args.label_dict,
                                         args.task_name)
            train_dataset = tokenize_dataset(train_dataset, tokenizer)
        else:
            raise NotImplementedError(f"sample_from: {args.sample_from}")

        return analysis_dataset, train_dataset, demonstration
    def prepare_analysis_mydataset(seed):
        demonstration, train_samples = sample_two_set_with_shot_per_class(dataset_trian,
                                                                          args.demonstration_shot,
                                                                          args.train_num_per_class,
                                                                          seed,
                                                                          label_name='label',
                                                                          a_total_shot=args.demonstration_total_shot)

        test_sample =dataset_test
        dataset_statistics(test_sample['label'])
        analysis_dataset = wrap_dataset(test_sample, demonstration, args.label_dict,
                                        args.task_name)
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        train_dataset = wrap_dataset(train_samples, demonstration, args.label_dict,
                                     args.task_name)
        train_dataset = tokenize_dataset(train_dataset, tokenizer)
        return analysis_dataset, train_dataset, demonstration

    def prepare_analysis_define_domo():

        # test_sample =dataset_test
        dataset_statistics(dataset_trian['labels'])
        analysis_dataset =dataset_trian
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        return analysis_dataset
    ys = []
    for seed in args.seeds:
        print(seed,'\n')
        if args.sample_from=='mydataset':
            analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(                seed)
        elif args.sample_from=='mydataset_define_demo':
            analysis_dataset =prepare_analysis_define_domo()
        else:
            analysis_dataset, train_dataset, demonstration = prepare_analysis_dataset(            seed)

        s_train=train_dataset[0]
        print(analysis_dataset[0]['sentence'])
        training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                          per_gpu_eval_batch_size=1,
                                          per_gpu_train_batch_size=1)
        trainer = Trainer(model=model, args=training_args)

        num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)
        predictor = Predictor(label_dict=args.label_dict,label_id_dict=args.label_id_dict, pad_token_id=tokenizer.pad_token_id,
                              task_name=args.task_name, tokenizer=tokenizer, layer=num_layer)
        if args.model_name in ['gpt2-xl']:
            attentionermanger = GPT2AttentionerManager(model.model, len(demonstration),
                                                       predictor=predictor,
                                                       device=model.device,n_class_tokens=n_class_tokens, n_head = args.n_head)
        elif 'llama-2' in args.model_name.lower():
            attentionermanger = Llama2AttentionerManager(model.model, len(demonstration),
                                                       predictor=predictor,
                                                       device=model.device,n_class_tokens=n_class_tokens, n_head=args.n_head)
        elif 'llama-3' in args.model_name.lower():
            attentionermanger = Llama2AttentionerManager(model.model, len(demonstration),
                                                       predictor=predictor,
                                                       device=model.device,n_class_tokens=n_class_tokens, n_head=args.n_head)
        else:
            raise NotImplementedError(f"model_name: {args.model_name}")

        params = attentionermanger.params()
        optimizer = Adam(params, lr=args.lr)

        set_seed(seed)
        loss_list = []
        for epoch in tqdm(range(args.epoch_num)):
            loss_item = 0.
            train_dataset = train_dataset.shuffle()
            train_dataloader = trainer.get_eval_dataloader(train_dataset)
            for idx, data in enumerate(train_dataloader):
                data = dict_to(data, model.device)  #set to cuda
                output = model(**data)
                label = data['labels']
                loss = F.cross_entropy(output['logits'], label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item += loss.item()
            loss_list.append(loss_item / idx)
            average_loss = float(loss_item / idx)
            print(f'{average_loss}/{epoch}')

        y = trainer.predict(analysis_dataset, ignore_keys=['results'])

        for _ in attentionermanger.attention_adapters:
            _.use_flag = False
        y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])

        # ys.append((y,loss_list, params, y2, average_loss))
        ys.append((y,y2))

    with open(save_file, 'wb') as f:
        pickle.dump([ys, ], f)




def notrain(train_file,test_file,multi_token,tokenizer, model, args: ReweightingArgs):

    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from=='mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)
    elif args.sample_from == 'mydataset_define_demo':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        # dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        # dataset_test = Dataset.from_pandas(dftest)

    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    # model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)
    n_class_tokens=0
    for k in args.label_id_dict_all:
        n_class_tokens+=len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token,model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)
    print(args.label_dict)
    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)
    def prepare_analysis_define_domo():
        dataset_statistics(dataset_trian['labels'])
        analysis_dataset =dataset_trian
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        return analysis_dataset

    if args.sample_from=='mydataset':
        analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(42)
    elif args.sample_from=='mydataset_define_demo':
        analysis_dataset =prepare_analysis_define_domo()
    else:
        analysis_dataset, train_dataset, demonstration = prepare_analysis_dataset(42)
    print(analysis_dataset[0]['sentence'])
    labels = np.array(analysis_dataset['labels'])

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_device_eval_batch_size=1,
                                      per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=training_args)

    y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])

    acc_2,false_list = get_acc(y2, labels)
    print(acc_2)
    return acc_2,false_list

def notrain_valid(train_file,test_file,multi_token,tokenizer, model, args: ReweightingArgs):

    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from == 'mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)
    elif args.sample_from == 'mydataset_define_demo':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)

    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    # model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)
    n_class_tokens = 0
    for k in args.label_id_dict_all:
        n_class_tokens += len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token, model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)
    print(args.label_dict)
    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)

    def prepare_analysis_define_domo():
        dataset_statistics(dataset_trian['labels'])
        analysis_dataset = dataset_trian
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        return analysis_dataset

    if args.sample_from == 'mydataset':
        analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(42)
    elif args.sample_from == 'mydataset_define_demo':
        analysis_dataset = prepare_analysis_define_domo()
    else:
        analysis_dataset, train_dataset, demonstration = prepare_analysis_dataset(42)
    print(analysis_dataset[0]['sentence'])
    labels = np.array(analysis_dataset['labels'])

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_device_eval_batch_size=1,
                                      per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=training_args)

    y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])


    acc_2, false_list = get_acc(y2, labels)

    def cal_probs_tokens(logits, label_map_tokens):
        sorted_label_map_tokens = {key: label_map_tokens[key] for key in sorted(label_map_tokens, key=int)}

        for c in sorted_label_map_tokens:
            # interest_index = [sublist[0] for sublist in sorted_label_map_tokens[c]]
            interest_index = sorted_label_map_tokens[c][0]

            # logit_c = logits[:, interest_index].max(dim=1).values.unsqueeze(1)
            logit_c =logits[:,interest_index].unsqueeze(1)
            try:
                logits_c = torch.cat((logits_c, logit_c), dim=1)
            except:
                logits_c = logit_c
        probs = F.softmax(logits_c, dim=-1)

        return probs, logits_c

    def get_acc_verbalizer(y, labels, label_map_tokens):
        ori_logits = y.predictions[2]
        logit = torch.tensor(ori_logits)
        probs, logits = cal_probs_tokens(logit, label_map_tokens)
        logits = logits.numpy()
        acc = accuracy_score(labels, np.argmax(logits, axis=1))

        return acc

    label_map = {}
    for k, v in args.label_dict.items():
        label_map[k]=tokenizer.encode( v , add_special_tokens=False)[1:]


    acc_first_token = get_acc_verbalizer(y2, labels, label_map)

    # print(acc_2)
    return acc_first_token, acc_2
    #

def notrain_test(train_file, test_file, multi_token,save_file,tokenizer, model, args: ReweightingArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from == 'mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)
    elif args.sample_from == 'mydataset_define_demo':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)

    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    # model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)
    n_class_tokens = 0
    for k in args.label_id_dict_all:
        n_class_tokens += len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token, model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)
    print(args.label_dict)
    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)

    def prepare_analysis_define_domo():
        dataset_statistics(dataset_trian['labels'])
        analysis_dataset = dataset_trian
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        return analysis_dataset

    if args.sample_from == 'mydataset':
        analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(42)
    elif args.sample_from == 'mydataset_define_demo':
        analysis_dataset = prepare_analysis_define_domo()
    else:
        analysis_dataset, train_dataset, demonstration = prepare_analysis_dataset(42)
    print(analysis_dataset[0]['sentence'])
    labels = np.array(analysis_dataset['labels'])

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_device_eval_batch_size=1,
                                      per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=training_args)

    y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])

    # ys.append((y2))
    with open(save_file, 'wb') as f:
        pickle.dump([y2, ], f)

    acc_2, false_list = get_acc(y2, labels)

    def cal_probs_tokens(logits, label_map_tokens):
        sorted_label_map_tokens = {key: label_map_tokens[key] for key in sorted(label_map_tokens, key=int)}

        for c in sorted_label_map_tokens:
            # interest_index = [sublist[0] for sublist in sorted_label_map_tokens[c]]
            interest_index = sorted_label_map_tokens[c][0]

            # logit_c = logits[:, interest_index].max(dim=1).values.unsqueeze(1)
            logit_c =logits[:,interest_index].unsqueeze(1)
            try:
                logits_c = torch.cat((logits_c, logit_c), dim=1)
            except:
                logits_c = logit_c
        probs = F.softmax(logits_c, dim=-1)

        return probs, logits_c

    def get_acc_verbalizer(y, labels, label_map_tokens):
        ori_logits = y.predictions[2]
        logit = torch.tensor(ori_logits)
        probs, logits = cal_probs_tokens(logit, label_map_tokens)
        logits = logits.numpy()
        acc = accuracy_score(labels, np.argmax(logits, axis=1))

        return acc

    label_map = {}
    for k, v in args.label_dict.items():
        label_map[k]=tokenizer.encode( v , add_special_tokens=False)[1:]


    acc_first_token = get_acc_verbalizer(y2, labels, label_map)

    # print(acc_2)
    return acc_2, acc_first_token
    #


def notrain_rawtest(train_file, test_file, multi_token,save_file,tokenizer, model, args: ReweightingArgs):
    # if os.path.exists(args.save_file_name):
    #     return
    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from == 'mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)
    elif args.sample_from == 'mydataset_define_demo':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)

    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")

    # model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)
    n_class_tokens = 0
    for k in args.label_id_dict_all:
        n_class_tokens += len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token, model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)
    print(args.label_dict)
    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)

    def prepare_analysis_define_domo():
        dataset_statistics(dataset_trian['labels'])
        analysis_dataset = dataset_trian
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        return analysis_dataset

    if args.sample_from == 'mydataset':
        analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(42)
    elif args.sample_from == 'mydataset_define_demo':
        analysis_dataset = prepare_analysis_define_domo()
    else:
        analysis_dataset, train_dataset, demonstration = prepare_analysis_dataset(42)
    print(analysis_dataset[0]['sentence'])
    labels = np.array(analysis_dataset['labels'])

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_device_eval_batch_size=1,
                                      per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=training_args)

    y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])

    # ys.append((y2))
    with open(save_file, 'wb') as f:
        pickle.dump([y2, ], f)

    acc_2, false_list = get_acc(y2, labels)


    return acc_2, false_list
    #


def notrain_random_test(train_file,test_file,save_file,multi_token,args: ReweightingArgs):

    set_gpu(args.gpu)
    if args.sample_from == 'test':
        dataset = load_huggingface_dataset_train_and_test(args.task_name)
    elif args.sample_from=='mydataset':
        dftrain = pd.read_csv(train_file)  # replace with your actual path
        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_trian = Dataset.from_pandas(dftrain)
        dataset_test = Dataset.from_pandas(dftest)

    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")
    model, tokenizer = load_model_and_tokenizer(args)
    args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)
    args.label_id_dict_all = get_label_id_dict_for_args_all(args, tokenizer)

    n_class_tokens=0
    for k in args.label_id_dict_all:
        n_class_tokens+=len(args.label_id_dict_all[k])
    model.config.output_attentions = True
    model = LMForwardAPI(multi_token=multi_token,model=model, model_name=args.model_name, tokenizer=tokenizer,
                         device='cuda:0',
                         label_dict=args.label_dict)

    training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size)

    def dataset_statistics(label):
        count_0 = {}
        for key in label:
            count_0[key] = count_0.get(key, 0) + 1

        print(count_0)

    def prepare_analysis_mydataset(seed):
        demonstration, train_samples = sample_two_set_with_shot_per_class(dataset_trian,
                                                                          args.demonstration_shot,
                                                                          args.train_num_per_class,
                                                                          seed,
                                                                          label_name='label',
                                                                          a_total_shot=args.demonstration_total_shot)

        test_sample =dataset_test
        dataset_statistics(test_sample['label'])
        analysis_dataset = wrap_dataset(test_sample, demonstration, args.label_dict,
                                        args.task_name)
        analysis_dataset = tokenize_dataset(analysis_dataset, tokenizer)

        train_dataset = wrap_dataset(train_samples, demonstration, args.label_dict,
                                     args.task_name)
        train_dataset = tokenize_dataset(train_dataset, tokenizer)


        return analysis_dataset, train_dataset, demonstration


    for seed in args.seeds:
        print(seed,'\n')
        analysis_dataset, train_dataset, demonstration = prepare_analysis_mydataset(
            seed)
        print(analysis_dataset[0]['sentence'])
        labels = np.array(analysis_dataset['labels'])

        training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                          per_device_eval_batch_size=1,
                                          per_device_train_batch_size=1)
        trainer = Trainer(model=model, args=training_args)

        y2 = trainer.predict(analysis_dataset, ignore_keys=['results'])

        with open(save_file, 'wb') as f:
            pickle.dump([y2, ], f)
        acc_2,false_list = get_acc(y2, labels)
        # print(acc_2)
        return acc_2,false_list


def get_label_dict(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for k, v in label_dicts.items():
        label_id_dict_all[k]=[]
        for w in v:
            label_id_dict_all[k].append(tokenizer.encode(w, add_special_tokens=False)[1:])
    return label_id_dict_all

def get_label_dict_gpt2xl(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for k, v in label_dicts.items():
        label_id_dict_all[k]=[]
        for w in v:
            label_id_dict_all[k].append(tokenizer.encode(w, add_special_tokens=False)[0:])
    return label_id_dict_all

def get_label_dict_llama3(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for k, v in label_dicts.items():
        label_id_dict_all[k]=[]
        for w in v:
            label_id_dict_all[k].append(tokenizer.encode(w, add_special_tokens=False)[0:])
    return label_id_dict_all


def get_label_first_token(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for k, v in label_dicts.items():
        label_id_dict_all[k]=[]
        for w in v:
            label_id_dict_all[k].append(tokenizer.encode(w, add_special_tokens=False)[1])


    return label_id_dict_all

def get_verbalizer_dict(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for k, v in label_dicts.items():
        k=int(k)
        label_id_dict_all[k]= {}
        for w in v:
            label_id_dict_all[k][w]=tokenizer.encode(w, add_special_tokens=False)[1:]
    return label_id_dict_all
def get_verbalizer_dict_gpt(label_dicts, tokenizer, args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all = {}
    for k, v in label_dicts.items():
        k = int(k)
        label_id_dict_all[k] = {}
        for w in v:
            label_id_dict_all[k][w] = tokenizer.encode(w, add_special_tokens=False)[0:]

    return label_id_dict_all

def get_verbalizer_dict_llama3(label_dicts, tokenizer, args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all = {}
    for k, v in label_dicts.items():
        k = int(k)
        label_id_dict_all[k] = {}
        for w in v:
            label_id_dict_all[k][w] = tokenizer.encode(w, add_special_tokens=False)[0:]

    return label_id_dict_all


def get_listverbalizer_dict(label_dicts,tokenizer,args: ReweightingArgs):
    set_gpu(args.gpu)

    label_id_dict_all={}
    for w in label_dicts:
        label_id_dict_all[w]=tokenizer.encode(w, add_special_tokens=False)[1:]

    return label_id_dict_all