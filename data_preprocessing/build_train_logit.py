import csv
import json
# model_name='gpt2-xl'
model_name='llama3'
for task_name in [
    'imdb',
                       'sst2',
                       'cr',
                       'aman',
                       'isear',
                       'trec',
    'ag_news'
                       ]:
    # Path to the input CSV file
    input_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/results/logitbased/zsl//train_{task_name}_logit_method2_{model_name}.csv'
    output_file = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_method2_{model_name}.csv'
    output_file2 = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_method2_{model_name}.json'


    # Dictionary to store the maximum score entry for each label
    max_scores = {}

    # Read the input CSV file
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            score = float(row['score'])  # Convert score to float for comparison
            # Check if this label is not in the dictionary or this score is higher than the current max
            if label not in max_scores or score > max_scores[label]['score']:
                max_scores[label] = {'text': row['text'], 'label': label, 'score': score}
    new_dict={}
    for key in max_scores:
        new_dict[max_scores[key]['label']]=max_scores[key]['score']
    sorted_dict = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_label=[int(key) for key in sorted_dict]
    with open(output_file2, 'w') as sfile:
        json.dump(sorted_label, sfile)
    # Write the maximum score entries to a new CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in max_scores.values():
            writer.writerow(entry)
