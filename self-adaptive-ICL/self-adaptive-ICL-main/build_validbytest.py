# import os
# os.environ['BNB_CUDA_VERSION'] = '117'
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
# os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import csv
import pandas as pd
from datasets import Dataset

task_name ='isear'

test_dict = {}
rows = [['text', 'label', 'idx']]

'''valid'''
retrieve_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/output/{task_name}/meta-llama/Llama-2-7b-chat-hf/1/test/selectvalid/retrieved.json'
dftrain = pd.read_csv(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data//train_{task_name}.csv')  # replace with your actual path
savefilevalid=f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/valid/valid_{task_name}.csv'
savefiletrain=f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/train/train_{task_name}.csv'
dataset_trian = Dataset.from_pandas(dftrain)
with open(retrieve_file, 'r') as file:
    # Load the data from the file
    verb_dict = json.load(file)
valid_set=[]
for eg in verb_dict:
    if eg['ctxs'][0] not in valid_set:
        valid_set.append(eg['ctxs'][0])
        if eg['text'] not in test_dict:
            test_dict[eg['text']] = {'class': eg['label'], 'text': [dataset_trian[eg['ctxs'][0]]['text']],
                                     'label': [dataset_trian[eg['ctxs'][0]]['label']]}
        else:
            test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs'][0]]['text'])
            test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs'][0]]['label'])

for j, test in enumerate(test_dict):
    row = []

    row.append(test_dict[test]['text'][0])
    row.append(test_dict[test]['label'][0])
    # row.append(inputs)
    row.append(j)
    rows.append(row)



with open(savefilevalid, mode='w') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
train_rows=[['text', 'label', 'idx']]
countt=0

for i in range(len(dataset_trian)):
    if i not in valid_set:
        row=[]
        row.append(dataset_trian[i]['text'])
        row.append(dataset_trian[i]['label'])
        row.append(countt)
        countt+=1
        train_rows.append(row)

print(f'valid: {len(rows)}, origin train: {len(dataset_trian)}, fileter_train: {len(train_rows)}\n residual: {len(dataset_trian)-len(train_rows)-len(rows)}')
with open(savefiletrain, mode='w') as tfile:
    writer = csv.writer(tfile)
    writer.writerows(train_rows)