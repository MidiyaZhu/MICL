import json
import csv
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
# task_name='isear'
# task_name='aman'
root = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/'
for task_name in ['isear','aman']:
    if task_name=='isear':
        label_dict={'fear':0,'sadness':1,'disgust':2,'anger':3,'joy':4,'guilt':5,'shame':6}
        map_dict= {0:'fear',1:'sadness',2:'disgust',3:'anger',4:'joy',5:'guilt',6:'shame'}
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
        map_dict= {0:'fear',1:'sadness',2:'disgust',3:'anger',4:'joy',5:'surprise',6:'others'}

    num_ice = 8
    num_candidates = 30
    prerank_method = 'topk'
    score_method = 'mdl'
    model_name = 'gpt2-xl'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
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
    run_dir = os.path.join(root, f'output/{task_name}/{model_name}/{rand_seed}/{dataset_split}')
    # rows=[['test','text','label']]
    # rows={}
    rows=[['text','labels','sentence','idx']]
    '''mdl'''
    # if 'llama' in model_name and score_method == 'mdl':
    #     trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_MDL_llama'
    # else:
    #     trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_MDL_{model_name}'
    # #
    if 'llama' in model_name and score_method == 'mdl':
        trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_topk_llama'
    else:
        trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_topk_{model_name}'

    test_dict={}
    # for label in isear_label_dict:
    for label in label_dict:
        retrieve_file = os.path.join(run_dir, f'retrieved2{label}.json')
        # retrieve_file2 = os.path.join(run_dir, 'retrieved2.json')
        dftrain = pd.read_csv(
            f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/train_{task_name}_{label}.csv')  # replace with your actual path

        dataset_trian = Dataset.from_pandas(dftrain)
        with open(retrieve_file,
                'r') as file:
            # Load the data from the file
            verb_dict = json.load(file)
        for eg in verb_dict:
            if eg['text'] not in test_dict:
                test_dict[eg['text']]={'class':eg['label'],'text':[dataset_trian[eg['ctxs_candidates'][0][0]]['text']],'label':[dataset_trian[eg['ctxs_candidates'][0][0]]['label']]}
            else:
                test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs_candidates'][0][0]]['text'])
                test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs_candidates'][0][0]]['label'])


            # if eg['text'] not in test_dict:
            #     test_dict[eg['text']]={'class':eg['label'],'text':[dataset_trian[eg['ctxs'][0]]['text']],'label':[dataset_trian[eg['ctxs'][0]]['label']]}
            # else:
            #     test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs'][0]]['text'])
            #     test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs'][0]]['label'])

    solotest_dicts = {}
    for test in test_dict:
        if test_dict[test]['label']!=[0,1,2,3,4,5,6]:
            print(test)
            print(test_dict[test]['label'])
        else:
            solotest_dicts[test]=test_dict[test]
    # for j in range(len(solotest_dicts)):
    format_s_dict = {
        'sst2': 'Review: {text}\nSentiment:{label}',
        'cr': 'Review: {text}\nSentiment:{label}',
        'agnews': 'Article: {text}\nAnswer:{label}',
        'trec': 'Question: {question}\nAnswer Type:{label}',
        'emo': 'Dialogue: {text}\nEmotion:{label}',
        'isear': 'Review: {text}\nEmotion:{label}',
        'aman': 'Review: {text}\nEmotion:{label}',
    }
    format_s = format_s_dict[task_name]
    for j, test in enumerate(solotest_dicts):
        row=[]
        prompts = [format_s.format(text=solotest_dicts[test]['text'][i], label=map_dict[solotest_dicts[test]['label'][i]] ) for
                   i in range(len(solotest_dicts[test]['text']))]
        inputs = format_s.format(text=test, label="")

        if len(prompts) > 0:
            inputs = "\n".join(prompts + [inputs])
        row.append(test)
        row.append( solotest_dicts[test]['class'])
        row.append(inputs)
        row.append(j)
        rows.append(row)



    # test=solotest_dicts[j]
    # rows[j] = {}
    # rows[j]['text'] = test
    # rows[j]['label'] = solotest_dicts[test]['class']
    # for i in range(len(solotest_dicts[test]['text'])):
    #     # row.append(test)
    #     if 'demo' in rows[j]:
    #         rows[j]['demo'].append(solotest_dicts[test]['text'][i])
    #         rows[j]['demolabel'].append(solotest_dicts[test]['label'][i])
    #     else:
    #         rows[j]['demo']=[solotest_dicts[test]['text'][i]]
    #         rows[j]['label']=[solotest_dicts[test]['label'][i]]
    #     # row.append( solotest_dicts[test]['label'][i])
    #     # rows.append(row)

    with open(f'{trainfile}.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
# with open(f'{trainfile}.json', 'w') as file:
#     json.dump(rows, file, indent=4)
'''topk'''
#
# if 'llama' in model_name and prerank_method == 'topk':
#     trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_topk_llama.csv'
# else:
#     trainfile = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/{task_name}_topk_{model_name}.csv'
# test_dict={}
# for label in isear_label_dict:
# # for label in aman_label_dict:
#     retrieve_file = os.path.join(run_dir, f'retrieved2{label}.json')
#     # retrieve_file2 = os.path.join(run_dir, 'retrieved2.json')
#     dftrain = pd.read_csv(
#         f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/data/train_{task_name}_{label}.csv')  # replace with your actual path
#
#     dataset_trian = Dataset.from_pandas(dftrain)
#     with open(retrieve_file,
#             'r') as file:
#         # Load the data from the file
#         verb_dict = json.load(file)
#     for eg in verb_dict:
#         if eg['text'] not in test_dict:
#             test_dict[eg['text']]={'text':[dataset_trian[eg['ctxs_candidates'][0][0]]['text']],'label':[dataset_trian[eg['ctxs_candidates'][0][0]]['label']]}
#         else:
#             test_dict[eg['text']]['text'].append(dataset_trian[eg['ctxs_candidates'][0][0]]['text'])
#             test_dict[eg['text']]['label'].append(dataset_trian[eg['ctxs_candidates'][0][0]]['label'])
#
# solotest_dicts = {}
# for test in test_dict:
#     if test_dict[test]['label']!=[0,1,2,3,4,5,6]:
#         print(test)
#         print(test_dict[test]['label'])
#     else:
#         solotest_dicts[test]=test_dict[test]
# for test in solotest_dicts:
#     for i in range(len(solotest_dicts[test]['text'])):
#         row=[]
#         row.append(test)
#         row.append( solotest_dicts[test]['text'][i])
#         row.append( solotest_dicts[test]['label'][i])
#         rows.append(row)
#
# with open(trainfile, mode='w') as file:
#     writer = csv.writer(file)
#     writer.writerows(rows)