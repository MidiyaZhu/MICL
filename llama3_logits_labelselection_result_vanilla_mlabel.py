import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import get_label_dict,get_label_dict_gpt2xl,get_label_dict_llama3, ReweightingArgs
from transformers.hf_argparser import HfArgumentParser
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer
import numpy as np
import pandas as pd
from datasets import Dataset
import torch
import torch.nn.functional as F
import json

def load_jsonresult(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
def get_acc(y,labels):
    scores = y.predictions[0]
    acc = accuracy_score(labels, np.argmax(scores, axis=1))

    return acc


def get_confusion_matrix(y,labels):
    scores = y.predictions[0]
    confusionmatrix = confusion_matrix(labels, np.argmax(scores, axis=1))
    return confusionmatrix

def get_acc_verbalizer(y,labels,label_map_tokens):
    ori_logits=y.predictions[2]
    logit=torch.tensor(ori_logits)
    probs, logits = cal_probs_tokens(logit,label_map_tokens)
    logits=logits.numpy()
    acc = accuracy_score(labels, np.argmax(logits, axis=1))

    return acc

def get_acc_verbalizer_gpt2xl(y,labels,label_map_tokens):
    ori_logits=y.predictions[2]
    logit=torch.tensor(ori_logits)
    probs, logits = cal_probs_tokens_gpt2xl(logit,label_map_tokens)
    logits=logits.numpy()
    acc = accuracy_score(labels, np.argmax(logits, axis=1))

    return acc

def get_acc_verbalizer_llama3(y,labels,label_map_tokens):
    ori_logits=y.predictions[2]
    logit=torch.tensor(ori_logits)
    probs, logits = cal_probs_tokens_gpt2xl(logit,label_map_tokens)
    logits=logits.numpy()
    acc = accuracy_score(labels, np.argmax(logits, axis=1))

    return acc

def get_acc_multitokens(y,labels,label_map_tokens,method):
    ori_logits = y.predictions[2]
    logit = torch.tensor(ori_logits)
    probs, logits, d_idx = cal_probs_tokens(logit, label_map_tokens, method)
    logits = logits.numpy()
    acc = accuracy_score(labels, np.argmax(logits, axis=1))
    r = recall_score(labels, np.argmax(logits, axis=1), average='weighted')
    p = precision_score(labels, np.argmax(logits, axis=1), average='weighted')
    f1 = f1_score(labels, np.argmax(logits, axis=1), average='weighted')
    return acc, r, p, f1, d_idx

def cal_probs_tokens_gpt2xl(logits,label_map_tokens):
    sorted_label_map_tokens = {key: label_map_tokens[key] for key in sorted(label_map_tokens, key=int)}

    for c in sorted_label_map_tokens:
        interest_index = [sublist[0] for sublist in sorted_label_map_tokens[c]]

        logit_c = logits[:, interest_index].max(dim=1).values.unsqueeze(1)
        try:
            logits_c = torch.cat((logits_c, logit_c), dim=1)
        except:
            logits_c = logit_c
    probs = F.softmax(logits_c, dim=-1)

    return probs, logits_c
def cal_probs_tokens(logits,label_map_tokens):
    sorted_label_map_tokens = {key: label_map_tokens[key] for key in sorted(label_map_tokens, key=int)}

    for c in sorted_label_map_tokens:
        interest_index = [sublist[0] for sublist in sorted_label_map_tokens[c]]

        logit_c = logits[:, interest_index].max(dim=1).values.unsqueeze(1)
        try:
            logits_c = torch.cat((logits_c, logit_c), dim=1)
        except:
            logits_c = logit_c
    probs = F.softmax(logits_c, dim=-1)

    return probs, logits_c

def load_result(save_file_name):
    with open(save_file_name, 'rb') as f:
        return pickle.load(f)


parser = HfArgumentParser((ReweightingArgs,))
args, = parser.parse_args_into_dataclasses()
args.sample_from = 'mydataset_define_demo'
multi_token = True
args.demonstration_shot = 1
args.split=False

# args.model_name ='llama-2-7b-chat-hf'
# args.model_name = 'gpt2-xl'
args.model_name = 'Meta-Llama-3-8B'
if 'gpt2' in args.model_name:
    args.n_head = 25
else:
    args.n_head = 32
model, tokenizer = load_model_and_tokenizer(args)
for method in ['vanilla']:#
    for args.task_name in [
        'sst2',
        # 'cr',
        # 'imdb',
        # 'trec',
        # 'aman',
        #  'isear',
        # 'ag_news'
    ]:#'sst2','trec',,'emo'
        if method in ['topk','selficl','vanilla',] and args.task_name=='imdb':
            test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test/test_{args.task_name}_mdl.csv'
        else:
            test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test/test_{args.task_name}.csv'

        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_test = Dataset.from_pandas(dftest)
        test_sample=dataset_test

        '''origin'''
        if args.task_name == 'isear':
            map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'], 6: [' shame']}
            args.label_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'], 6: [' shame']}
        elif args.task_name == 'aman':
            map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' surprise'],   6: [' neutral']}
            args.label_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' surprise'],    6: [' neutral']}
        elif args.task_name == 'trec':
            # map_dict =  {0: [" abbreviation"], 1: [" animal"], 2: [" definition"], 3: [" persons"], 4: [" state"],5: [" numeric"]}
            # args.label_dict = {0: [" abbreviation"], 1: [" animal"], 2: [" definition"], 3: [" persons"], 4: [" state"], 5: [" numeric"]}
            map_dict = {  0: [' abbreviation'],        1: [' entity'],        2:[ ' description'],        3:[ ' human'],        4:[ ' location'],        5: [' number']  }
            args.label_dict ={  0: [' abbreviation'],        1: [' entity'],        2:[ ' description'],        3:[ ' human'],        4:[ ' location'],        5: [' number']  }
        elif args.task_name == 'cr' or args.task_name == 'sst2' or args.task_name == 'imdb':
            map_dict = {0: [' negative'], 1: [' positive']}
            args.label_dict = {0: [' negative'], 1: [' positive']}
        elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
            args.label_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
            map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
        accall, acc_verbalzer_acc = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}, {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        for seed in [42,43,44,45,46]:
            epoch=1
            if 'gpt' in args.model_name.lower():
                verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_{args.model_name}.json')
                save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/gpt2xl/new/test/select_{args.task_name}_{epoch}_test_{seed}'
                raw_label_map = get_label_dict_gpt2xl(args.label_dict, tokenizer, args)
            elif 'llama-3' in args.model_name.lower():
                verbalizer = load_jsonresult( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json')
                save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama3/test/select_{args.task_name}_{epoch}_test_{seed}'
                raw_label_map = get_label_dict_llama3(args.label_dict, tokenizer, args)
            else:
                verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb.json')
                save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama7b/test/select_{args.task_name}_{epoch}_test_{seed}'
                raw_label_map = get_label_dict(args.label_dict, tokenizer, args)

            labels = np.array(test_sample['label'])

            results = load_result(save_file)

            y2 = results[0]

            # acc_2 = get_acc_verbalizer(y2, labels, label_map)
            acc = get_acc(y2, labels)

            if 'gpt' in args.model_name:
                acc_2 = get_acc_verbalizer_gpt2xl(y2, labels, raw_label_map)
                acc_label = get_acc_verbalizer_gpt2xl(y2, labels, raw_label_map)
            elif 'llama-3' in args.model_name.lower():
                acc_2 = get_acc_verbalizer_llama3(y2, labels, raw_label_map)
                acc_label = get_acc_verbalizer_llama3(y2, labels, raw_label_map)
            else:
                acc_2 = get_acc_verbalizer(y2, labels, raw_label_map)
                acc_label = get_acc_verbalizer(y2, labels, raw_label_map)
            acc_verbalzer_acc[epoch].append(acc_2)
            accall[epoch].append(acc_label)

            # print(f'{args.task_name}-{epoch}_{method}:\n'
            #       # f'origin_acc= {round(acc*100,2)}\n'
            #       f'rawlabel_acc= {round(acc_label*100,2)}\n'
            #       f'verbalizer_select_acc= {round(acc_2*100,2)}\n'            #
            #       )

            for epoch in range(2,7):

                if 'gpt' in args.model_name:
                    map_dict = load_jsonresult( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/gpt2xl/new/{args.task_name}_logiticl_mapdict_{epoch}_{seed}.json')
                    # print(map_dict)
                    label_map = get_label_dict_gpt2xl(map_dict, tokenizer, args)
                    save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/gpt2xl/new/test/select_{args.task_name}_{epoch}_test_{seed}'
                elif 'llama-3' in args.model_name.lower():
                    map_dict = load_jsonresult( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama3//{args.task_name}_logiticl_mapdict_{epoch}_{seed}.json')
                    # print(map_dict)
                    label_map = get_label_dict_gpt2xl(map_dict, tokenizer, args)
                    save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama3//test/select_{args.task_name}_{epoch}_test_{seed}'

                else:
                    map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama7b/{args.task_name}_logiticl_mapdict_{epoch}_{seed}.json')
                    # print(map_dict)
                    label_map = get_label_dict(map_dict,tokenizer, args)
                    save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/{method}/llama7b/test/select_{args.task_name}_{epoch}_test_{seed}'

                labels = np.array(test_sample['label'])

                results = load_result(save_file)
                if len(results)==5:
                    y,y1,_,y2,y3 = results
                elif len(results)==6:
                    y, y1, _, y2, y3,label_map_tokens = results
                elif len(results) == 4:
                    y,y2, _, label_map_tokens = results
                elif len(results) == 3:
                    y,y2, label_map_tokens = results
                elif len(results) == 2:
                    y, y2 = results
                elif len(results) == 1:
                     y2 = results[0]
                if 'gpt' in args.model_name:
                    acc_2 = get_acc_verbalizer_gpt2xl(y2, labels, label_map)
                    acc_label = get_acc_verbalizer_gpt2xl(y2, labels, raw_label_map)
                else:
                    acc_2 = get_acc_verbalizer(y2, labels, label_map)
                    acc_label = get_acc_verbalizer(y2, labels, raw_label_map)
                acc = get_acc(y2, labels)
                acc_verbalzer_acc[epoch].append(acc_2)
                accall[epoch].append(acc_label)

                # print(f'{args.task_name}-{epoch}_{method}:\n'
                #       # f'origin_acc= {round(acc*100,2)}\n'
                #       f'rawlabel_acc= {round(acc_label*100,2)}\n'
                #       f'verbalizer_select_acc= {round(acc_2*100,2)}\n'
                #       )
        for epoch in accall:
            print(f'{args.task_name}-{epoch}_{method}:\n'
                  # f'origin_acc= {round(acc*100,2)}\n' 
                  f'rawlabel_acc ave= {round(np.mean(accall[epoch]) * 100, 2)}\n'
                  f'verbalizer_select_acc= {round(np.mean(acc_verbalzer_acc[epoch]) * 100, 2)}\n'
                  # f'acc_verbalzer_acc:{acc_verbalzer_acc[epoch]}'
                  )
