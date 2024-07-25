import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import get_verbalizer_dict,get_label_dict_gpt2xl, get_verbalizer_dict_llama3, notrain_rawtest,ReweightingArgs
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

def build_prompt_datasettest(args,label_dict,map_dict,savefile):
    test_dict = {}
    rows = [['text', 'labels', 'sentence', 'idx']]
    for label in label_dict:
        '''valid'''
        if args.task_name=='ag_news':
            retrieve_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/output/{args.task_name}/test/1000/retrieved2{label}.json'
            dftrain = pd.read_csv(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{args.task_name}_1000_{label}.csv')  # replace with your actual path

        else:
            retrieve_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/output/{args.task_name}/test/retrieved2{label}.json'
            dftrain = pd.read_csv(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{args.task_name}_{label}.csv')  # replace with your actual path

        dataset_trian = Dataset.from_pandas(dftrain)
        with open(retrieve_file, 'r') as file:
            # Load the data from the file
            verb_dict = json.load(file)
        for eg in verb_dict:
            if eg['text'] not in test_dict:
                test_dict[eg['text']] = {'class': eg['label'], 'text': [dataset_trian[eg['ctxs'][0]]['text']],
                                         'label': [dataset_trian[eg['ctxs'][0]]['label']]}
            else:
                if dataset_trian[eg['ctxs'][0]]['label'] not in test_dict[eg['text']]['label']:
                    test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs'][0]]['text'])
                    test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs'][0]]['label'])

    solotest_dicts = {}
    for test in test_dict:
        if args.task_name in ['aman', 'isear']:
            if test_dict[test]['label'] != [0, 1, 2, 3, 4, 5, 6]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['cr', 'sst2', 'imdb']:
            if test_dict[test]['label'] != [0, 1]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['trec']:
            if test_dict[test]['label'] != [0, 1, 2, 3, 4, 5]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['ag_news']:
            if test_dict[test]['label'] != [0, 1, 2, 3]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
    # for j in range(len(solotest_dicts)):
    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    for j, test in enumerate(solotest_dicts):
        row = []
        if args.task_name == 'imdb':
            prompts = [format_s.format(text=test_dict[test]['text'][i][:2900],
                                       label=map_dict[int(test_dict[test]['label'][i])]) for i in
                       range(len(test_dict[test]['text']))]
        else:
            prompts = [format_s.format(text=solotest_dicts[test]['text'][i],
                                       label=map_dict[int(solotest_dicts[test]['label'][i])]) for i in
                       range(len(solotest_dicts[test]['text']))]
        inputs = format_s.format(text=test, label="")

        if len(prompts) > 0:
            inputs = "\n".join(prompts + [inputs])
        row.append(test)
        row.append(solotest_dicts[test]['class'])
        row.append(inputs)
        row.append(j)
        rows.append(row)

    with open(savefile, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def build_prompt_datasetvalid(args,map_dict,savefile):
    test_dict = {}
    rows = [['text', 'labels', 'sentence', 'idx']]
    for label in label_dict:
        '''valid'''
        retrieve_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/output/{args.task_name}/valid/1/retrieved2{label}.json'
        dftrain = pd.read_csv(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{args.task_name}_{label}.csv')  # replace with your actual path

        dataset_trian = Dataset.from_pandas(dftrain)
        with open(retrieve_file, 'r') as file:
            # Load the data from the file
            verb_dict = json.load(file)
        for eg in verb_dict:
            if eg['text'] not in test_dict:
                test_dict[eg['text']] = {'class': eg['label'], 'text': [dataset_trian[eg['ctxs'][0]]['text']],
                                         'label': [dataset_trian[eg['ctxs'][0]]['label']]}
            else:
                if dataset_trian[eg['ctxs'][0]]['label'] not in test_dict[eg['text']]['label']:
                    test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs'][0]]['text'])
                    test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs'][0]]['label'])

    solotest_dicts = {}
    for test in test_dict:
        if args.task_name in ['aman', 'isear']:
            if test_dict[test]['label'] != [0, 1, 2, 3, 4, 5, 6]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['cr', 'sst2', 'imdb']:
            if test_dict[test]['label'] != [0, 1]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['trec']:
            if test_dict[test]['label'] != [0, 1, 2, 3, 4, 5]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
        elif args.task_name in ['ag_news']:
            if test_dict[test]['label'] != [0, 1, 2, 3]:
                print(test)
                print(test_dict[test]['label'])
            else:
                solotest_dicts[test] = test_dict[test]
    # for j in range(len(solotest_dicts)):
    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'imdb': 'Review: {text}\nSentiment:{label}',
        'ag_news': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {text}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[args.task_name]
    for j, test in enumerate(solotest_dicts):
        row = []
        if args.task_name == 'imdb':
            prompts = [format_s.format(text=test_dict[test]['text'][i][:2900],
                                       label=map_dict[int(test_dict[test]['label'][i])]) for i in
                       range(len(test_dict[test]['text']))]
        else:
            prompts = [format_s.format(text=solotest_dicts[test]['text'][i],
                                       label=map_dict[int(solotest_dicts[test]['label'][i])]) for i in
                       range(len(solotest_dicts[test]['text']))]
        inputs = format_s.format(text=test, label="")

        if len(prompts) > 0:
            inputs = "\n".join(prompts + [inputs])
        row.append(test)
        row.append(solotest_dicts[test]['class'])
        row.append(inputs)
        row.append(j)
        rows.append(row)

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
for args.task_name in [
    'aman', 'trec',
    'cr',
    'sst2', 'isear',
    'ag_news', 'imdb'
]:
    '''origin'''
    if args.task_name == 'isear':
        label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
        args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt', 6: ' shame'}
        map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'],
                    6: [' shame']}
    elif args.task_name == 'aman':
        label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'others': 6}
        args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise',
                           6: ' others'}
        map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' surprise'],
                    6: [' others']}
    elif args.task_name == 'trec':
        label_dict = {'abbreviation': 0, 'entity': 1, 'description': 2, 'human': 3, 'location': 4, 'number': 5}
        args.label_dict = {0: ' abbreviation', 1: ' entity', 2: ' description', 3: ' human', 4: ' location',
                           5: ' number'}
        map_dict = {0: [' abbreviation'], 1: [' entity'], 2: [' description'], 3: [' human'], 4: [' location'],
                    5: [' number']}
    elif args.task_name == 'cr' or args.task_name == 'sst2' or args.task_name == 'imdb' or args.task_name == 'glue-sst2100':
        label_dict = {'negative': 0, 'positive': 1}
        args.label_dict = {0: ' negative', 1: ' positive'}
        map_dict = {0: [' negative'], 1: [' positive']}
    elif args.task_name == 'ag_news100' or args.task_name == 'ag_news':
        label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
        args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
        map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
    if args.task_name in [ 'aman']:
        epochs= range(1,7)
    else:
        epochs= range(1,7)
    for epoch in epochs:
        test_csv_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/test_{args.task_name}.csv'

        map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/{args.task_name}_logiticl_mapdict_{epoch}.json')
        candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
        train_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/mselficl_{args.task_name}_{epoch}_test_llama3.csv'
        build_prompt_datasettest(args, label_dict, candidate_map_dict, train_save_path)
        select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/test/select_{args.task_name}_{epoch}_test'
        acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer, model, args)

        '''valid'''
        train_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/selflogiticlselect_{args.task_name}_{epoch}_valid_llama3.csv'
        map_dict = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/{args.task_name}_logiticl_mapdict_{epoch}.json')
        candidate_map_dict = {int(k): ''.join(v) for k, v in map_dict.items()}
        build_prompt_datasetvalid(args, candidate_map_dict, train_save_path)
        select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/valid/select_{args.task_name}_{epoch}_valid'
        acc, false_list_2 = evaluate(train_save_path, test_csv_path, multi_token, select_save_path, tokenizer, model,args)

        raw_verbalizer = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json')
        better_label = load_jsonresult(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/{args.task_name}_logiticl_mapdict_{epoch}.json')
        verbalizer={key:[] for key in raw_verbalizer}
        for key in raw_verbalizer:
            for w in raw_verbalizer[key]:
                if w not in better_label[key]:
                    verbalizer[key].append(w)


        save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/valid/select_{args.task_name}_{epoch}_valid'
        test_file =  f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/sota/valid_selficl_{args.task_name}.csv'



        all_results = []

        dftest = pd.read_csv(test_file)  # replace with your actual path
        dataset_test = Dataset.from_pandas(dftest)

        test_sample=dataset_test

        if 'gpt' in args.model_name.lower():
            label_map_verbalizer = get_label_dict_gpt2xl(verbalizer, tokenizer, args)
        elif 'llama-3' in args.model_name.lower():
            label_map_verbalizer = get_verbalizer_dict_llama3(verbalizer, tokenizer, args)
        else:
            label_map_verbalizer = get_verbalizer_dict(verbalizer, tokenizer, args)

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

        with open(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/1shot/labelselect/selficl/llama3/{args.task_name}_logiticl_mapdict_{epoch+1}.json','w') as file:
            json.dump(better_label, file)

