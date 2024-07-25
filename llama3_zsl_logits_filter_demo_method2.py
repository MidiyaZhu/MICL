import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import get_verbalizer_dict,get_verbalizer_dict_gpt,get_verbalizer_dict_llama3, ReweightingArgs
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
from scipy.stats import pointbiserialr
import copy
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


def get_logits_verbalizer(labels,ori_logits,label_map_tokens):
    verbalizer_logits={}

    logit=torch.tensor(ori_logits)
    for i in range(len(logit)):
        verbalizer_logits[i] = {key: {} for key in label_map_tokens.keys()}

        for k in label_map_tokens:
            for w in label_map_tokens[k]:
                verbalizer_logits[i][k][w] = logit[i][label_map_tokens[k][w][0]]

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

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def load_logit(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        # Create a CSV reader object using the file object
        reader = csv.DictReader(file)
        # next(reader)
        verblogit={}
        for row in reader:
            verblogit[row['word']]=row['logits']
    return verblogit


parser = HfArgumentParser((ReweightingArgs,))
args, = parser.parse_args_into_dataclasses()
# args.model_name ='llama-2-7b-chat-hf'
# args.model_name = 'gpt2-xl'
args.model_name = 'Meta-Llama-3-8B'
if 'gpt2' in args.model_name:
    args.n_head=25
else:
    args.n_head=32
args.sample_from = 'mydataset_define_demo'
multi_token = True
args.demonstration_shot = 1
args.split=False
method = 'MDL'  # MDL topk




model, tokenizer = load_model_and_tokenizer(args)
for args.task_name in [
    # 'imdb',
    #                    'sst2',
    #                    'cr',
    #                    'aman',
    #                    'isear',
    #                    'trec',
    'ag_news'
]:

    if args.task_name == 'isear':
        label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'guilt': 5, 'shame': 6}
        args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' guilt', 6: ' shame'}
        raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' guilt'], 6: [' shame']}
    elif args.task_name == 'aman':
        label_dict = {'fear': 0, 'sadness': 1, 'disgust': 2, 'anger': 3, 'joy': 4, 'surprise': 5, 'others': 6}
        args.label_dict = {0: ' fear', 1: ' sadness', 2: ' disgust', 3: ' anger', 4: ' joy', 5: ' surprise', 6: ' others'}
        raw_map_dict = {0: [' fear'], 1: [' sadness'], 2: [' disgust'], 3: [' anger'], 4: [' joy'], 5: [' surprise'], 6: [' others']}
    elif args.task_name == 'trec':
        label_dict = {'abbreviation': 0, 'entity': 1, 'description': 2, 'human': 3, 'location': 4, 'number': 5}
        args.label_dict = {0: ' abbreviation', 1: ' entity', 2: ' description', 3: ' human', 4: ' location', 5: ' number'}
        raw_map_dict = {0: [' abbreviation'], 1: [' entity'], 2: [' description'], 3: [' human'], 4: [' location'],5: [' number']}
    elif args.task_name == 'cr' or args.task_name == 'sst2'  or args.task_name == 'imdb':
        label_dict = {'negative': 0, 'positive': 1}
        args.label_dict = {0: ' negative', 1: ' positive'}
        raw_map_dict = {0: [' negative'], 1: [' positive']}
    elif args.task_name == 'ag_news100' or  args.task_name == 'ag_news':
        label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
        args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
        raw_map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}
    if args.task_name in ['trec','aman']:
        verbalizerfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json'
        verbalizers = load_json(verbalizerfile)
        verb_logit_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_llama3_movelogit.csv'
    else:
        verbalizerfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_movelogit_rpb_llama3.json'
        verbalizers = load_json(verbalizerfile)
        verb_logit_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{args.task_name}_verbalizer_llama3_movelogit.csv'
    verb_logit_weight = load_logit(verb_logit_file)
    # 计算点双列相关系数
    for labelname in label_dict:
        if args.task_name == 'ag_news':
            test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{args.task_name}_1000_{labelname}.csv'
        else:
            test_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{args.task_name}_1_{labelname}.csv'

        all_results = []
        if args.task_name in ['emo']:
            dataset = load_huggingface_dataset_train_and_test(args.task_name)
        elif args.task_name in ['isear', 'aman', 'cr', 'sst2', 'trec', 'imdb', 'ag_news']:
            dftest = pd.read_csv(test_file)  # replace with your actual path
            dataset_test = Dataset.from_pandas(dftest)

            test_sample = dataset_test
            # elif task_name == 'emo':
        else:
            test_sample = dataset['test'].shuffle(seed=seed).select(range(min(1000, len(dataset['test']))))

        save_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/{args.task_name}_train_{labelname}_llama3'

        try:
            fulltest_sample = pd.concat([fulltest_sample, dftest], ignore_index=True)
        except:
            fulltest_sample = dftest
        if 'gpt' in args.model_name:
            label_map_origin = get_verbalizer_dict_gpt(verbalizers, tokenizer, args)
        elif 'llama-2' in args.model_name:
            label_map_origin = get_verbalizer_dict(verbalizers, tokenizer, args)
        else:
            label_map_origin = get_verbalizer_dict_llama3(verbalizers, tokenizer, args)
        label_tokens,label_words,label_words_label=[],[],[]
        for l in label_map_origin:
            for w in label_map_origin[l]:
                label_tokens.append(label_map_origin[l][w][0])
                label_words.append(w)
                label_words_label.append(l)
        labels = np.array(test_sample['label'])
        try:
            fulllabel = np.concatenate((fulllabel, labels))
        except:
            fulllabel = np.copy(labels)

        results = load_result(save_file)
        if len(results) == 5:
            y, y1, _, y2, y3 = results
        elif len(results) == 6:
            y, y1, _, y2, y3, label_map_tokens = results
        elif len(results) == 4:
            y, y2, _, label_map_tokens = results
        elif len(results) == 3:
            y, y2, label_map_tokens = results
        elif len(results) == 2:
            y, y2 = results
        elif len(results) == 1:
            y2 = results[0]
        logits = y2.predictions[2]
        try:
            fulllogits = np.concatenate((fulllogits, logits))
        except:
            fulllogits = np.copy(logits)
    test_sample = Dataset.from_pandas(fulltest_sample)

    test_logit=fulllogits[:, label_tokens]
    sorted_indices = np.argsort(-test_logit, axis=1)

    # If you need to sort the actual values using these indices (optional)
    sorted_values = np.take_along_axis(test_logit, sorted_indices, axis=1)
    output_array = np.array(label_words_label)[sorted_indices]
    rows=[['text','label','score']]
    count,f=0,0
    data_sta={}
    countlabel={key:0 for key in  args.label_dict}
    for i in range(len(output_array)):

        if output_array[i][0]==test_sample[i]['label']:
            countlabel[test_sample[i]['label']]+=1
            row=[]
            row.append(test_sample[i]['text'])
            row.append(test_sample[i]['label'])
            score=0
            vl=len(verbalizers[str(test_sample[i]['label'])])
            for v in range(vl):
                if output_array[i][v]==test_sample[i]['label']:
                    labelword=label_words[sorted_indices[i][v]]
                    logit_weight=float(verb_logit_weight[labelword])
                    if logit_weight>0:
                        s=logit_weight
                    else:
                        s=0
                else:
                    s=0
                score+=s
            row.append(score)
            # row.append(count)
            rows.append(row)
            count+=1
            if test_sample[i]['label'] not in data_sta:
                data_sta[test_sample[i]['label']] = 1
            else:
                data_sta[test_sample[i]['label']] += 1
        else:
            f+=1
    '''if no samples meets the requirements'''
    for labelcheck in countlabel:
        if countlabel[labelcheck]==0:
            for i in range(len(output_array)):
                row = []
                row.append(test_sample[i]['text'])
                row.append(test_sample[i]['label'])
                score = 0
                vl = len(verbalizers[str(test_sample[i]['label'])])
                for v in range(vl):
                    if output_array[i][v] == test_sample[i]['label']:
                        labelword = label_words[sorted_indices[i][v]]
                        logit_weight = float(verb_logit_weight[labelword])
                        if logit_weight > 0:
                            s = logit_weight
                        else:
                            s = 0
                    else:
                        s = 0
                    score += s
                row.append(score)
                # row.append(count)
                rows.append(row)
                count += 1
                if test_sample[i]['label'] not in data_sta:
                    data_sta[test_sample[i]['label']] = 1
                else:
                    data_sta[test_sample[i]['label']] += 1


    print(f)
    print(count)


    for k in data_sta:
        print(f'{k}:{data_sta[k]}\n')
    if args.task_name in ['trec','aman']:
        filter_demo_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/train_{args.task_name}_logit_method2_llama3.csv'
    else:
        filter_demo_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/train_{args.task_name}_logit_method2_llama3.csv'

    with open(filter_demo_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)



