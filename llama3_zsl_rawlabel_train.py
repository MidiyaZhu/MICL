import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from icl.analysis.reweightingwithacc import notrain_rawtest, ReweightingArgs
from transformers.hf_argparser import HfArgumentParser
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer
import random
import json
import csv
import pandas as pd
from datasets import Dataset
import ast
from copy import deepcopy


def load_result(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def evaluate(train_csv_path,test_csv_path,multi_token,select_save_path,tokenizer, model,args):
    if args.split == False:
        acc,false_list=notrain_rawtest(train_csv_path, test_csv_path, multi_token,select_save_path,tokenizer, model, args)
    return acc,false_list

parser = HfArgumentParser((ReweightingArgs,))
args, = parser.parse_args_into_dataclasses()
# args.model_name = 'llama-2-7b-chat-hf'
args.model_name = 'Meta-Llama-3-8B'
if 'gpt2' in args.model_name:
    args.n_head=25
else:
    args.n_head=32

args.sample_from = 'mydataset_define_demo'
multi_token = True

args.demonstration_shot = 1
args.split = False
model, tokenizer = load_model_and_tokenizer(args)

for args.task_name in [
    'sst2',
    'aman',
    # 'cr', 'isear', 'trec',
# 'imdb',
#     'ag_news'

                        ]:#

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
        label_dict = {'negative': 0,'positive': 1}#
        args.label_dict = {0: ' negative', 1: ' positive'}
        raw_map_dict = {0: [' negative'], 1: [' positive']}
    elif args.task_name == 'ag_news100' or  args.task_name == 'ag_news':
        label_dict = {'Worlds': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
        args.label_dict = {0: ' Worlds', 1: ' Sports', 2: ' Business', 3: ' Technology'}
        raw_map_dict = {0: [' Worlds'], 1: [' Sports'], 2: [' Business'], 3: [' Technology']}

    for labelname in label_dict:
        if args.task_name=='ag_news':
            valid_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{args.task_name}_1000_{labelname}.csv'
        else:
            valid_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{args.task_name}_1_{labelname}.csv'


        savefile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/temp/{args.task_name}train{labelname}.csv'

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
        rows = [['text', 'labels', 'sentence', 'idx']]
        with open(valid_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            count=0
            for row in csv_reader:
                text = row[0]
                label = row[1]
                idx=count
                row = []

                inputs = format_s.format(text=text, label="")

                row.append(text)
                row.append(label)
                row.append(inputs)
                row.append(idx)
                rows.append(row)
                count+=1

        with open(savefile, mode='w') as file:
            writer = csv.writer(file)
            writer.writerows(rows)


        select_save_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl/{args.task_name}_train_{labelname}_llama3'

        acc, false_list_2 = evaluate(savefile, valid_file, multi_token, select_save_path, tokenizer, model, args)

