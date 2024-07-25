import csv
from collections import defaultdict


# Function to split and save CSV files based on the label
def split_and_save_csv_valid(task_name,label_dict,filename,seed):
    # Use defaultdict to store rows by labels
    rows_by_label = defaultdict(list)
    label_counters = defaultdict(int)
    # Read the CSV file and split rows by label
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']

            row['idx'] = label_counters[label]
            label_counters[label] += 1
            label = row['label']
            rows_by_label[label].append(row)

    # Write the split rows into separate CSV files
    for label, rows in rows_by_label.items():
        new_filename = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{task_name}_{seed}_{label_dict[int(label)]}.csv'
        with open(new_filename, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def split_and_save_csv_test(task_name,label_dict,filename):
    # Use defaultdict to store rows by labels
    rows_by_label = defaultdict(list)
    label_counters = defaultdict(int)
    # Read the CSV file and split rows by label
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']

            row['idx'] = label_counters[label]
            label_counters[label] += 1
            label = row['label']
            rows_by_label[label].append(row)
    # Write the split rows into separate CSV files
    for label, rows in rows_by_label.items():
        new_filename = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{task_name}_{label_dict[int(label)]}.csv'
        with open(new_filename, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
for task_name in ['ag_news','imdb']:#'isear','aman','cr','trec',
    '''valid'''
    for seed in [1]:
        # Assuming the file to read is 'train.csv'
        if task_name=='trec':
            label_dict = {
              0:  'abbreviation',
               1: 'entity',
                2:'description',
                3:'human',
                4:'location',
                5:'number'
            }
        elif task_name in ['cr','sst2','imdb']:
            label_dict = {
              0:  'negative',
               1: 'positive',
            }
        elif task_name=='aman':
            label_dict = {
              0:  'fear',
               1: 'sadness',
                2:'disgust',
                3:'anger',
                4:'joy',
                5:'surprise',
                6: 'others'
            }
        elif task_name=='isear':
            label_dict = {
              0:  'fear',
               1: 'sadness',
                2:'disgust',
                3:'anger',
                4:'joy',
                5:'guilt',
                6: 'shame'
            }
        elif task_name == 'ag_news':
            label_dict = {
                0: 'Worlds',
                1: 'Sports',
                2: 'Business',
                3: 'Technology',

            }
        savedfile=f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{task_name}_{seed}.csv'

        split_and_save_csv_valid(task_name,label_dict,savedfile,seed)
    '''test'''


    if task_name=='trec':
        label_dict = {
          0:  'abbreviation',
           1: 'entity',
            2:'description',
            3:'human',
            4:'location',
            5:'number'
        }
    elif task_name=='ag_news':
        label_dict = {
          0:  'Worlds',
           1: 'Sports',
            2:'Business',
            3:'Technology',

        }
    elif task_name in ['cr','sst2','imdb']:
        label_dict = {
          0:  'negative',
           1: 'positive',
        }
    elif task_name=='aman':
        label_dict = {
          0:  'fear',
           1: 'sadness',
            2:'disgust',
            3:'anger',
            4:'joy',
            5:'surprise',
            6: 'others'
        }
    elif task_name=='isear':
        label_dict = {
          0:  'fear',
           1: 'sadness',
            2:'disgust',
            3:'anger',
            4:'joy',
            5:'guilt',
            6: 'shame'
        }

    savedfile=f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/train/train_{task_name}.csv'

    split_and_save_csv_test(task_name,label_dict,savedfile)
