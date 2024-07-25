import csv
# model_name='gpt2-xl'
model_name='llama3'
for method in ['method1','method2']:
    for task_name in [
        'aman',
        'trec',
        'cr',
        'sst2',
        'imdb',
        'isear',
        'ag_news'
                           ]:
        if task_name=='ag_news':
            file2_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{task_name}_1000.csv'
        else:
            file2_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/train_{task_name}_1.csv'
        file1_path = f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/train_{task_name}_{method}_{model_name}.csv'
        output_path =  f'/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/label-words-are-anchors-main/data/new/logitbased/train/logit_score/valid_{task_name}_{method}_{model_name}.csv'

        # Step 1: Read the first file and collect the values of the first column
        first_column_values = set()
        with open(file1_path, mode='r', newline='') as file1:
            reader = csv.reader(file1)
            for row in reader:
                first_column_values.add(row[0])
        print(f'origin: {len(first_column_values)}')

        # Step 2: Read the second file and write out rows that don't have matching first column in the set
        with open(file2_path, mode='r', newline='') as file2, open(output_path, mode='w', newline='') as outfile:
            reader = csv.reader(file2)
            writer = csv.writer(outfile)
            writer.writerow( ['text', 'label', 'idx'])
            c=0
            a=0
            for row in reader:
                c+=1
                if row[0] not in first_column_values:
                    writer.writerow(row)
                    a+=1
        print(f'demo: {c}\nvalid:{a}')

        print("File has been processed and saved as", output_path)
