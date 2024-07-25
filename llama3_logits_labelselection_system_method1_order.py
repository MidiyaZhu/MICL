import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import get_verbalizer_dict,get_verbalizer_dict_gpt,get_verbalizer_dict_llama3, notrain_rawtest,ReweightingArgs
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
from collections import defaultdict

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

def evaluate(train_csv_path,test_csv_path,multi_token,select_save_path,tokenizer, model,args):
    if args.split == False:
        acc,false_list=notrain_rawtest(train_csv_path, test_csv_path, multi_token,select_save_path,tokenizer, model, args)
    return acc,false_list
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
def build_prompt_datasettest_noorder(args,label_dict,map_dict, method,savefile):
    if args.task_name in ['trec','aman']:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    else:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    dtext, dlabel = [], []
    with open(demofile, mode='r', encoding='utf-8') as demo_file:
        demo_reader = csv.reader(demo_file)
        next(demo_reader)
        for drow in demo_reader:
            dtext.append(drow[0])
            dlabel.append(int(drow[1]))

    valid_file =f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test/test_{args.task_name}.csv'
    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'glue-sst2100': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'ag_news100': 'Article: {text}\nAnswer:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    rows = [['text', 'labels', 'sentence', 'idx']]
    with open(valid_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        count = 0
        for row in csv_reader:
            text = row[0]
            label = row[1]
            idx = count
            row = []
            if args.task_name == 'imdb':
                prompts = [format_s.format(text=dtext[i][:2500],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            else:
                prompts = [format_s.format(text=dtext[i],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            inputs = format_s.format(text=text, label="")

            if len(prompts) > 0:
                inputs = "\n".join(prompts + [inputs])

            # inputs = format_s.format(text=text, label="")

            row.append(text)
            row.append(int(label))
            row.append(inputs)
            row.append(idx)
            rows.append(row)
            count += 1

    with open(savefile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def build_prompt_datasetvalid_noorder(args,map_dict,savefile,method):
    if args.task_name in ['trec','aman']:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    else:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    dtext, dlabel = [], []
    with open(demofile, mode='r', encoding='utf-8') as demo_file:
        demo_reader = csv.reader(demo_file)
        next(demo_reader)
        for drow in demo_reader:
            dtext.append(drow[0])
            dlabel.append(int(drow[1]))

    if args.task_name in ['trec','aman']:
        valid_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'
    else:
        valid_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'

    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'glue-sst2100': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'ag_news100': 'Article: {text}\nAnswer:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    rows = [['text', 'labels', 'sentence', 'idx']]
    with open(valid_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        count = 0
        for row in csv_reader:
            text = row[0]
            label = row[1]
            idx = count
            row = []
            if args.task_name == 'imdb':
                prompts = [format_s.format(text=dtext[i][:2500],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            else:
                prompts = [format_s.format(text=dtext[i],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            inputs = format_s.format(text=text, label="")

            if len(prompts) > 0:
                inputs = "\n".join(prompts + [inputs])

            # inputs = format_s.format(text=text, label="")

            row.append(text)
            row.append(int(label))
            row.append(inputs)
            row.append(idx)
            rows.append(row)
            count += 1

    with open(savefile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
def build_prompt_datasettest(args,label_dict,map_dict,method,savefile):
    test_dict = {}
    rows = [['text', 'labels', 'sentence', 'idx']]
    if args.task_name in ['trec','aman']:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    else:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'

    dtext, dtext_tmp, dlabel, dlabel_tmp = [], [], [], []
    with open(demofile, mode='r', encoding='utf-8') as demo_file:
        demo_reader = csv.reader(demo_file)
        next(demo_reader)
        for drow in demo_reader:
            dtext_tmp.append(drow[0])
            dlabel_tmp.append(int(drow[1]))
    for ll in label_dict:
        dtext.append(dtext_tmp[dlabel_tmp.index(label_dict[ll])])
        dlabel.append(dlabel_tmp[dlabel_tmp.index(label_dict[ll])])
    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    valid_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/test/test_{args.task_name}.csv'
    with open(valid_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        count = 0
        for row in csv_reader:
            text = row[0]
            label = row[1]
            idx = count
            row = []
            if args.task_name == 'imdb':
                prompts = [format_s.format(text=dtext[i][:2500],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            else:
                prompts = [format_s.format(text=dtext[i],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            inputs = format_s.format(text=text, label="")

            if len(prompts) > 0:
                inputs = "\n".join(prompts + [inputs])

            row.append(text)
            row.append(int(label))
            row.append(inputs)
            row.append(idx)
            rows.append(row)
            count += 1

    with open(savefile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def build_prompt_datasetvalid(args,label_dict,map_dict,savefile,method):
    if args.task_name in ['trec','aman']:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    else:
        demofile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.csv'
    dtext, dtext_tmp, dlabel, dlabel_tmp = [], [], [], []
    with open(demofile, mode='r', encoding='utf-8') as demo_file:
        demo_reader = csv.reader(demo_file)
        next(demo_reader)
        for drow in demo_reader:
            dtext_tmp.append(drow[0])
            dlabel_tmp.append(int(drow[1]))
    for ll in label_dict:
        dtext.append(dtext_tmp[dlabel_tmp.index(label_dict[ll])])
        dlabel.append(dlabel_tmp[dlabel_tmp.index(label_dict[ll])])
    if args.task_name in ['trec','aman']:
        valid_file =f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'
    else:
        valid_file =f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'

    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'glue-sst2100': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'ag_news100': 'Article: {text}\nAnswer:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    rows = [['text', 'labels', 'sentence', 'idx']]
    with open(valid_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        count = 0
        for row in csv_reader:
            text = row[0]
            label = row[1]
            idx = count
            row = []
            if args.task_name == 'imdb':
                prompts = [format_s.format(text=dtext[i][:2500],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            else:
                prompts = [format_s.format(text=dtext[i],label=map_dict[int(dlabel[i])]) for i in range(len(dtext))]
            inputs = format_s.format(text=text, label="")

            if len(prompts) > 0:
                inputs = "\n".join(prompts + [inputs])

            # inputs = format_s.format(text=text, label="")

            row.append(text)
            row.append(int(label))
            row.append(inputs)
            row.append(idx)
            rows.append(row)
            count += 1

    with open(savefile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
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
for method in ['method1']:  #,'method2'
    if method=='method1':
        tasknames=['cr','aman','trec','sst2','isear','imdb','ag_news']
    elif method=='method2':
        tasknames=['cr','aman','trec','sst2','isear','imdb','ag_news']#'imdb''aman','isear','trec',
    for args.task_name in tasknames:

        if args.task_name == 'cr':
            epochs = range(2, 7)
        else:
            epochs = range(1, 7)
        for epoch in epochs:


            if args.task_name == 'isear':
                label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
                args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt',
                                   6: ' shame'}
                argslabel_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt',
                                  6: ' shame'}
                raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
                                5: [' guilt'],
                                6: [' shame']}
            elif args.task_name == 'aman':
                label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'noemotion': 6}
                args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
                                   6: ' noemotion'}
                argslabel_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
                                  6: ' noemotion'}
                raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
                                5: [' surprise'], 6: [' noemotion']}
            elif args.task_name == 'trec':
                label_dict = {'abbreviation': 0, 'element': 1, 'definition': 2, 'individual': 3, 'place': 4, 'number': 5}
                args.label_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place',  5: ' number'}
                argslabel_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place',  5: ' number'}
                raw_map_dict = {0: [" abbreviation"], 1: [" element"], 2: [" definition"], 3: [" individual"],
                                4: [" place"], 5: [" number"]}
            elif args.task_name == 'cr' or args.task_name == 'sst2' or args.task_name == 'imdb':
                label_dict = {'negative': 0, 'positive': 1}
                args.label_dict = {0: ' negative', 1: ' positive'}
                argslabel_dict = {0: ' negative', 1: ' positive'}
                raw_map_dict = {0: [' negative'], 1: [' positive']}
            elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
                label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
                args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
                argslabel_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
                map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
            test_csv_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test_{args.task_name}.csv'
            '''test'''

            select_key_list = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.json')
            select_label_dict = {argslabel_dict[int(key)].strip(): int(key) for key in select_key_list}
            if epoch==1:
                map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
            else:
                map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_order_{epoch}_llama3.json')
            candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
            train_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/iticlelect_{args.task_name}_{epoch}_test_order.csv'
            build_prompt_datasettest(args, select_label_dict, candidate_map_dict, method, train_save_path)
            select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/test/select_{args.task_name}_{epoch}_test_{method}_order_llama3'
            acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer,model, args)

            '''valid'''
            select_key_list = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{args.task_name}_{method}_llama3.json')
            select_label_dict = {argslabel_dict[int(key)].strip(): int(key) for key in select_key_list}
            if epoch == 1:
                map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
            else:
                map_dict = load_jsonresult( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_order_{epoch}_llama3.json')
            candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
            train_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/iticlelect_{args.task_name}_{epoch + 1}_valid_order.csv'

            build_prompt_datasetvalid(args, select_label_dict, candidate_map_dict, train_save_path,method)
            select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/valid/select_{args.task_name}_{epoch}_valid_{method}_order_llama3'
            acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer, model,args)


            raw_verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json')
            if epoch == 1:
                better_label = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
            else:
                better_label = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_order_{epoch}_llama3.json')
            verbalizer={key:[] for key in raw_verbalizer}
            for key in raw_verbalizer:
                for w in raw_verbalizer[key]:
                    if w not in better_label[key]:
                        verbalizer[key].append(w)

            save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/valid/select_{args.task_name}_{epoch}_valid_{method}_order_llama3'
            test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'

            all_results = []

            dftest = pd.read_csv(test_file)  # replace with your actual path
            dataset_test = Dataset.from_pandas(dftest)

            test_sample=dataset_test
            '''origin'''
            if args.task_name == 'isear':
                label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
                args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt', 6: ' shame'}
                map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'], 6: [' shame']}
            elif args.task_name == 'aman':
                label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'noemotion': 6}
                args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',6: ' noemotion'}
                map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
                                5: [' surprise'], 6: [' noemotion']}
            elif args.task_name == 'trec':
                label_dict = {'abbreviation': 0, 'element': 1, 'definition': 2, 'individual': 3, 'place': 4, 'number': 5}
                args.label_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place', 5: ' number'}
                raw_map_dict = {0: [" abbreviation"], 1: [" element"], 2: [" definition"], 3: [" individual"], 4: [" place"], 5: [" number"]}
            elif args.task_name == 'cr' or args.task_name == 'sst2'  or args.task_name == 'imdb' or args.task_name=='glue-sst2100':
                label_dict = {'negative': 0, 'positive': 1}
                args.label_dict = {0: ' negative', 1: ' positive'}
                map_dict = {0: [' negative'], 1: [' positive']}
            elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
                label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
                args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
                map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}

            if 'gpt' in args.model_name.lower():
                label_map_verbalizer = get_verbalizer_dict_gpt(verbalizer, tokenizer, args)
            elif 'llama-2' in args.model_name.lower():
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
            logits_verbalizer = get_verbalizer_logits(y2, labels,label_map_verbalizer)
            label_logits_verbalizer=[]
            for k in range(len(logits_verbalizer)):
                tem_v = []
                for v in range(len(logits_verbalizer[k])):
                    tem_logits_v=[]
                    for l in args.label_dict:
                        tem_label_logit = []
                        for t in range(len(logits_verbalizer[k][v])):
                            if labels[t]==l:
                                tem_label_logit.append(logits_verbalizer[k][v][t])
                        tem_logits_v.append(np.mean(tem_label_logit))
                    tem_v.append(tem_logits_v)
                label_logits_verbalizer.append(tem_v)
            '''evaluate the quality of the words'''
            pass_label_logits_verbalizer=[]
            pass_label_logits_verbalizer_word=[]
            for l in range(len(label_logits_verbalizer)):
                tem,tem_word=[],[]
                for w in range(len(label_logits_verbalizer[l])):
                    if l==label_logits_verbalizer[l][w].index(max(label_logits_verbalizer[l][w])):
                        tem.append(label_logits_verbalizer[l][w][l])
                        tem_word.append(verbalizer[str(l)][w])
                pass_label_logits_verbalizer.append(tem)
                pass_label_logits_verbalizer_word.append(tem_word)
            for l in range(len(pass_label_logits_verbalizer)):
                if len(pass_label_logits_verbalizer[l])>0:
                    max_idx=pass_label_logits_verbalizer[l].index(max(pass_label_logits_verbalizer[l]))
                    better_label[str(l)].append(pass_label_logits_verbalizer_word[l][max_idx])

            with open(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_order_{epoch+1}_llama3.json','w') as file:
                json.dump(better_label, file)

    # '''no order'''
    # for args.task_name in tasknames:
    #
    #     if args.task_name == 'aman':
    #         epochs = range(1, 7)
    #     else:
    #         epochs = range(1, 7)
    #     for epoch in epochs:
    #
    #
    #         if args.task_name == 'isear':
    #             label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
    #             args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt',
    #                                6: ' shame'}
    #             argslabel_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt',
    #                               6: ' shame'}
    #             raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
    #                             5: [' guilt'],
    #                             6: [' shame']}
    #         elif args.task_name == 'aman':
    #             label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'noemotion': 6}
    #             args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
    #                                6: ' noemotion'}
    #             argslabel_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
    #                               6: ' noemotion'}
    #             raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
    #                             5: [' surprise'], 6: [' noemotion']}
    #         elif args.task_name == 'trec':
    #             label_dict = {'abbreviation': 0, 'element': 1, 'definition': 2, 'individual': 3, 'place': 4, 'number': 5}
    #             args.label_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place',  5: ' number'}
    #             argslabel_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place',  5: ' number'}
    #             raw_map_dict = {0: [" abbreviation"], 1: [" element"], 2: [" definition"], 3: [" individual"],
    #                             4: [" place"], 5: [" number"]}
    #         elif args.task_name == 'cr' or args.task_name == 'sst2' or args.task_name == 'imdb':
    #             label_dict = {'negative': 0, 'positive': 1}
    #             args.label_dict = {0: ' negative', 1: ' positive'}
    #             argslabel_dict = {0: ' negative', 1: ' positive'}
    #             raw_map_dict = {0: [' negative'], 1: [' positive']}
    #         elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
    #             label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
    #             args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
    #             argslabel_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
    #             map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
    #         test_csv_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test_{args.task_name}.csv'
    #         '''test'''
    #
    #
    #         if epoch==1:
    #             map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
    #         else:
    #             map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_{epoch}_llama3.json')
    #         candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
    #         train_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/iticl_{args.task_name}_{epoch}_test.csv'
    #         build_prompt_datasettest_noorder(args, label_dict, candidate_map_dict, method, train_save_path)
    #         select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3//test/select_{args.task_name}_{epoch}_test_{method}_llama3'
    #         acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer, model, args)
    #
    #         '''valid'''
    #
    #         if epoch == 1:
    #             map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
    #         else:
    #             map_dict = load_jsonresult( f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_{epoch}_llama3.json')
    #         candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
    #         build_prompt_datasetvalid_noorder(args, candidate_map_dict, train_save_path, method)
    #         select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/valid/select_{args.task_name}_{epoch}_valid_{method}_llama3'
    #         acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer, model, args)
    #
    #         raw_verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json')
    #         if epoch == 1:
    #             better_label = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{epoch}_llama3.json')
    #         else:
    #             better_label = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_{epoch}_llama3.json')
    #         verbalizer={key:[] for key in raw_verbalizer}
    #         for key in raw_verbalizer:
    #             for w in raw_verbalizer[key]:
    #                 if w not in better_label[key]:
    #                     verbalizer[key].append(w)
    #
    #         save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/valid/select_{args.task_name}_{epoch}_valid_{method}_llama3'
    #         test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{args.task_name}_{method}_llama3.csv'
    #
    #         all_results = []
    #
    #         dftest = pd.read_csv(test_file)  # replace with your actual path
    #         dataset_test = Dataset.from_pandas(dftest)
    #
    #         test_sample=dataset_test
    #         '''origin'''
    #         if args.task_name == 'isear':
    #             label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
    #             args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt', 6: ' shame'}
    #             map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'], 6: [' shame']}
    #         elif args.task_name == 'aman':
    #             label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'noemotion': 6}
    #             args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',6: ' noemotion'}
    #             map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'],
    #                             5: [' surprise'], 6: [' noemotion']}
    #         elif args.task_name == 'trec':
    #             label_dict = {'abbreviation': 0, 'element': 1, 'definition': 2, 'individual': 3, 'place': 4, 'number': 5}
    #             args.label_dict = {0: ' abbreviation', 1: ' element', 2: ' definition', 3: ' individual', 4: ' place', 5: ' number'}
    #             raw_map_dict = {0: [" abbreviation"], 1: [" element"], 2: [" definition"], 3: [" individual"], 4: [" place"], 5: [" number"]}
    #         elif args.task_name == 'cr' or args.task_name == 'sst2'  or args.task_name == 'imdb' or args.task_name=='glue-sst2100':
    #             label_dict = {'negative': 0, 'positive': 1}
    #             args.label_dict = {0: ' negative', 1: ' positive'}
    #             map_dict = {0: [' negative'], 1: [' positive']}
    #         elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
    #             label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
    #             args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
    #             map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
    #
            # if 'gpt' in args.model_name.lower():
            #     label_map_verbalizer = get_verbalizer_dict_gpt(verbalizer, tokenizer, args)
            # elif 'llama-2' in args.model_name.lower():
            #     label_map_verbalizer = get_verbalizer_dict(verbalizer, tokenizer, args)
            # else:
            #     label_map_verbalizer = get_verbalizer_dict_llama3(verbalizer, tokenizer, args)
    #         rows = [
    #             [['text', 'labels']],
    #             [['text', 'labels']],
    #             [['text', 'labels']],
    #             [['text', 'labels']],
    #             [['text', 'labels']],
    #             [['text', 'labels']],
    #             [['text', 'labels']]
    #         ]
    #
    #         for j in range(len(label_map_verbalizer)):
    #             for w in label_map_verbalizer[j]:
    #                 rows[j][0].append(w)
    #         labels = np.array(test_sample['label'])
    #
    #         results = load_result(save_file)
    #         if len(results)==5:
    #             y,y1,_,y2,y3 = results
    #         elif len(results)==6:
    #             y, y1, _, y2, y3,label_map_tokens = results
    #         elif len(results) == 4:
    #             y,y2, _, label_map_tokens = results
    #         elif len(results) == 3:
    #             y,y2, label_map_tokens = results
    #         elif len(results) == 2:
    #             y, y2 = results
    #         elif len(results) == 1:
    #             y2 = results[0]
    #
    #
    #         verbalizer_logits = get_logits_verbalizer(y2, labels,label_map_verbalizer)
    #         logits_verbalizer = get_verbalizer_logits(y2, labels,label_map_verbalizer)
    #         label_logits_verbalizer=[]
    #         for k in range(len(logits_verbalizer)):
    #             tem_v = []
    #             for v in range(len(logits_verbalizer[k])):
    #                 tem_logits_v=[]
    #                 for l in args.label_dict:
    #                     tem_label_logit = []
    #                     for t in range(len(logits_verbalizer[k][v])):
    #                         if labels[t]==l:
    #                             tem_label_logit.append(logits_verbalizer[k][v][t])
    #                     tem_logits_v.append(np.mean(tem_label_logit))
    #                 tem_v.append(tem_logits_v)
    #             label_logits_verbalizer.append(tem_v)
    #         '''evaluate the quality of the words'''
    #         pass_label_logits_verbalizer=[]
    #         pass_label_logits_verbalizer_word=[]
    #         for l in range(len(label_logits_verbalizer)):
    #             tem,tem_word=[],[]
    #             for w in range(len(label_logits_verbalizer[l])):
    #                 if l==label_logits_verbalizer[l][w].index(max(label_logits_verbalizer[l][w])):
    #                     tem.append(label_logits_verbalizer[l][w][l])
    #                     tem_word.append(verbalizer[str(l)][w])
    #             pass_label_logits_verbalizer.append(tem)
    #             pass_label_logits_verbalizer_word.append(tem_word)
    #         for l in range(len(pass_label_logits_verbalizer)):
    #             if len(pass_label_logits_verbalizer[l])>0:
    #                 max_idx=pass_label_logits_verbalizer[l].index(max(pass_label_logits_verbalizer[l]))
    #                 better_label[str(l)].append(pass_label_logits_verbalizer_word[l][max_idx])
    #
    #         with open(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/micl/llama3/{args.task_name}_logiticl_mapdict_{method}_{epoch+1}_llama3.json','w') as file:
    #             json.dump(better_label, file)
