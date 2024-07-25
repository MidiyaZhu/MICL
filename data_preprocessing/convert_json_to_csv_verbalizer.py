import pandas as pd
# model_name = 'gpt2-xl'
model_name = 'llama3'
for task_name in [
'cr',
    'sst2',
    'aman',
'isear',
    'trec',
    'imdb',

    'ag_news'

]:
    df = pd.read_csv(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{task_name}_verbalizer_{model_name}_movelogit.csv')

    # Create a dictionary where each label is a key and the words are list of values
    result = df.groupby('label')['word'].apply(list).to_dict()

    # Convert the dictionary to JSON
    import json
    with open(f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/verbalizer/basedonlogits/filter_low_quality/basedontrain/{task_name}_verbalizer_{model_name}_movelogit.json', 'w') as sfile:
        json.dump(result, sfile)
