import os
os.environ['BNB_CUDA_VERSION'] = '117'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/transformers'
os.environ['TORCH_HOME'] = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/cache/torch'
# Set environment variables
os.environ['WANDB_PROJECT'] = 'ICL'
os.environ['WANDB_ENTITY'] = 'zixiaozhu'
os.environ['WANDB_API_KEY'] = '08231f47d978105125589d4ff268a928417cd933'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HYDRA_FULL_ERROR'] = '1'
import subprocess

# Define variables
root = '/mnt/e72cc1a3-e45e-4a0c-b1e6-0e4357d2e753/zixiao/RAG/self-adaptive-ICL/self-adaptive-ICL-main/'
num_ice = 8
num_candidates = 30
prerank_method = 'topk'
score_method = 'mdl'
model_name = 'gpt2-xl'
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

# task_names = ['sst2', 'commonsense_qa', 'mnli', 'qnli']
task_names = ['trec']

for task_name in task_names:
    if task_name in ['commonsense_qa', 'mnli', 'qnli']:
        dataset_split = 'validation'
    else:
        dataset_split = 'test'

    if task_name in ['snli', 'mnli', 'qnli']:
        emb_field = 'ALL'
    else:
        emb_field = 'X'

    run_dir = os.path.join(root, f'output/{task_name}/{model_name}/{rand_seed}/{dataset_split}')
    retrieve_file = os.path.join(run_dir, 'retrieved.json')
    retrieve_file2 = os.path.join(run_dir, 'retrieved2.json')
    pred_file = os.path.join(run_dir, 'pred4.json')

    os.makedirs(run_dir, exist_ok=True)

    # Prerank step
    prerank_cmd = f"python prerank.py output_file={retrieve_file} emb_field={emb_field} num_ice={num_ice} method={prerank_method} num_candidates={num_candidates} dataset_reader.task_name={task_name} rand_seed={rand_seed} dataset_reader.dataset_split={dataset_split} index_reader.task_name={task_name} index_file={os.path.join(run_dir, 'index')} scale_factor=0.1"
    subprocess.run(prerank_cmd, shell=True, check=True)

    # Retriever step
    retriever_cmd = f"accelerate launch --num_processes {n_gpu} --main_process_port {port} retriever.py output_file={retrieve_file2} num_ice={num_ice} window={window} rand_seed={rand_seed} instruction_template={instruction_template} span={span} dataset_reader.task_name={task_name} dataset_reader.dataset_path={retrieve_file} batch_size={inf_batch_size} method={score_method}"
    subprocess.run(retriever_cmd, shell=True, check=True)

    # PPL Inference step
    ppl_inferencer_cmd = f"accelerate launch --num_processes {n_gpu} --main_process_port {port} ppl_inferencer.py dataset_reader.task_name={task_name} rand_seed={rand_seed} dataset_reader.dataset_path={retrieve_file2} instruction_template={instruction_template} span={span} dataset_reader.n_tokens={n_tokens} output_file={pred_file} model_name={model_name} batch_size={inf_batch_size}"
    subprocess.run(ppl_inferencer_cmd, shell=True, check=True)
