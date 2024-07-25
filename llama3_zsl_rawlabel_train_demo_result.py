import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import get_verbalizer_dict,get_verbalizer_dict_gpt,get_verbalizer_dict_llama3,get_label_dict, ReweightingArgs
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
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def load_jsonresult(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
def get_acc(y,labels):
    scores = y.predictions[0]
    acc = accuracy_score(labels, np.argmax(scores, axis=1))
    # r = recall_score(labels, np.argmax(scores, axis=1), average='weighted')
    # p = precision_score(labels, np.argmax(scores, axis=1), average='weighted')
    # f1 = f1_score(labels, np.argmax(scores, axis=1), average='weighted')
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


def get_logits_verbalizer(y,labels,label_map_tokens):
    verbalizer_logits={}
    ori_logits=y.predictions[2]
    logit=torch.tensor(ori_logits)
    for i in range(len(logit)):
        verbalizer_logits[i] ={key: {} for key in label_map_tokens.keys()}

        for k in label_map_tokens:
            for w in label_map_tokens[k]:
                verbalizer_logits[i][k][w]=logit[i][label_map_tokens[k][w][0]]


    return verbalizer_logits

def get_verbalizer_logits(y,labels,label_map_tokens):
    verbalizer_logits={}
    ori_logits=y.predictions[2]
    logit=torch.tensor(ori_logits)
    for k in label_map_tokens:
        verbalizer_logits[k]=[]
        count=0
        for w in label_map_tokens[k]:
            verbalizer_logits[k].append([])
            for i in range(len(logit)):
                verbalizer_logits[k][count].append(logit[i][label_map_tokens[k][w][0]].item())
            count+=1


    return verbalizer_logits


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
# args.model_name ='llama-2-7b-chat-hf'
# args.model_name = 'gpt2-xl'
args.model_name = 'Meta-Llama-3-8B'
if 'gpt2' in args.model_name:
    args.n_head = 25
else:
    args.n_head = 32

args.sample_from = 'mydataset_define_demo'
multi_token = True
args.demonstration_shot = 1
args.split=False




model, tokenizer = load_model_and_tokenizer(args)
for args.task_name in [
    # 'sst2',
    #     'aman',
    #     'cr',
    #     'isear',
    #     'trec',
    # 'imdb',
    'ag_news'
]:#'sst2','trec','agnews','emo'
    verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json')
    for method in ['method1']:#, 'method2'

        '''method2'''
        save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/demotrain/{args.task_name}_train_{method}_llama3'
        test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'

        all_results = []

        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_test = Dataset.from_pandas(dftest)

        test_sample=dataset_test
        '''origin'''
        if args.task_name == 'isear':
            label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
            args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt',
                               6: ' shame'}
            map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'],
                            6: [' shame']}
        elif args.task_name == 'aman':
            label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'neutral': 6}
            args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
                               6: ' neutral'}
            map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
                            5: [' surprise'], 6: [' neutral']}
        elif args.task_name == 'trec':
            label_dict = {'abbreviation': 0, 'entity': 1, 'description': 2, 'human': 3, 'location': 4, 'number': 5}
            args.label_dict = {0: ' abbreviation', 1: ' entity', 2: ' description', 3: ' human', 4: ' location',
                               5: ' number'}
            map_dict = {0: [' abbreviation'], 1: [' entity'], 2: [' description'], 3: [' human'], 4: [' location'],
                            5: [' number']}
        elif args.task_name == 'cr' or args.task_name == 'sst2'  or args.task_name == 'imdb' or args.task_name=='glue-sst2100':
            label_dict = {'negative': 0, 'positive': 1}
            args.label_dict = {0: ' negative', 1: ' positive'}
            map_dict = {0: [' negative'], 1: [' positive']}
        elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
            label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
            args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
            map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}

        if 'gpt' in args.model_name:
            label_map_verbalizer = get_verbalizer_dict_gpt(verbalizer, tokenizer, args)
        elif 'llama-2' in args.model_name:
            label_map_verbalizer = get_verbalizer_dict(verbalizer, tokenizer, args)
        else:
            label_map_verbalizer = get_verbalizer_dict_llama3(verbalizer, tokenizer, args)
        rows = [
            [['text', 'labels']],
            [['text', 'labels']],
            [['text', 'labels']],
            [['text', 'labels']],
            [['text', 'labels']],
            [['text', 'labels']],
            [['text', 'labels']]
        ]

        for j in range(len(label_map_verbalizer)):
            for w in label_map_verbalizer[j]:
                rows[j][0].append(w)
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


        verbalizer_logits = get_logits_verbalizer(y2, labels,label_map_verbalizer)
        text_label_verbalizer_logit={key: {k:0 for k in verbalizer_logits[key]} for key in verbalizer_logits}
        text_label_verbalizer_logit_word={key: {k:0 for k in verbalizer_logits[key]} for key in verbalizer_logits}
        for k in verbalizer_logits:
            for l in verbalizer_logits[k]:
                max_key = max(verbalizer_logits[k][l], key=verbalizer_logits[k][l].get)
                max_value = verbalizer_logits[k][l][max_key]
                text_label_verbalizer_logit[k][l]=max_value
                text_label_verbalizer_logit_word[k][l]=max_key
        #check whether the max logit in demo is in the correct label
        better_label={key:[] for key in text_label_verbalizer_logit}
        for l in text_label_verbalizer_logit:
            if l == max(text_label_verbalizer_logit[l], key=text_label_verbalizer_logit[l].get):
                better_label[l].append(text_label_verbalizer_logit_word[l][l])
            else:
                print(f'demo of label: {l} has wrong input-label mapping in verbalizer!!')

        with open( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/demotrain/{args.task_name}_logiticl_mapdict_1_{method}_llama3.json','w') as file:
        # with open( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/demotrain/{args.task_name}_logiticl_mapdict_1.json','w') as file:
            json.dump(better_label, file)
