import csv
import json
from collections import defaultdict
# model_name='gpt2-xl'
model_name='llama7b'
method='method1'
for task_name in [
    # 'imdb',
    #                    'sst2',
    #                    'cr',
    #                    'aman',
                       'isear',
                       # 'trec',
    'ag_news'
                       ]:
    # Path to the input CSV file
    if model_name=='gpt2-xl':
        if task_name=='trec':
            input_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl//train_{task_name}_logit_{method}_{model_name}2.csv'
        else:
            input_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl//train_{task_name}_logit_{method}_{model_name}.csv'
    else:
        if method=='{method}':
            input_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl//train_{task_name}_logit.csv'
        else:
            input_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl//train_{task_name}_logit_{method}.csv'
    output_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_{method}_{model_name}_5shot.csv'
    output_file3 = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_{method}_{model_name}_5shots.csv'
    output_file2 = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_{method}_{model_name}_5shot.json'

    # Dictionary to store the top-5 score entries for each label
    top_scores = defaultdict(list)

    # Read the input CSV file
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            score = float(row['score'])  # Convert score to float for comparison
            top_scores[label].append({'text': row['text'], 'label': label, 'score': score})

    # Keep only the top-5 scores for each label
    for label in top_scores:
        top_scores[label] = sorted(top_scores[label], key=lambda x: x['score'], reverse=True)[:5]

    # Flatten the dictionary for easy sorting
    flattened_scores = []
    for label, entries in top_scores.items():
        for entry in entries:
            flattened_scores.append(entry)

    # Sort all entries by score in descending order and get their labels
    sorted_scores = sorted(flattened_scores, key=lambda x: x['score'], reverse=True)
    sorted_labels = [int(entry['label']) for entry in sorted_scores]
    sort_label=[]
    for i in sorted_labels:
        if i not in sort_label:
            sort_label.append(i)
    # Write the sorted labels to a JSON file
    with open(output_file2, 'w') as sfile:
        json.dump(sort_label, sfile)

    # Write the top-5 score entries for each label to a new CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted_scores:
            writer.writerow(entry)

    data = []
    with open(output_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['score'] = float(row['score'])  # Convert score to float for comparison
            data.append(row)

    # Sort the data by score in descending order
    data.sort(key=lambda x: x['score'], reverse=True)

    # Reorder the data based on the specified label order [0, 3, 2, 1]
    label_order = sort_label
    reordered_data = []

    while data:
        for label in label_order:
            for i, row in enumerate(data):
                if int(row['label']) == label:
                    reordered_data.append(row)
                    data.pop(i)
                    break

    # Write the reordered data to a new CSV file
    with open(output_file3, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reordered_data:
            writer.writerow(row)