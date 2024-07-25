import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .load_local import load_local_model_or_tokenizer
from ..util_classes.arg_classes import DeepArgs
from transformers import BitsAndBytesConfig

def load_model_and_tokenizer(args: DeepArgs):
    if args.model_name in ['gpt2-xl', 'EleutherAI/gpt-j-6B', "llama-2-7b-chat-hf","llama-2-13b-chat-hf","meta-llama/Llama-2-7b-chat-hf","meta-llama/Meta-Llama-3-8B","Meta-Llama-3-8B"]:
        if args.model_name in ['llama-2-7b-chat-hf','meta-llama/Llama-2-7b-chat-hf']:
            args.model_name='meta-llama/Llama-2-7b-chat-hf'
            tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = load_local_model_or_tokenizer(args.model_name, 'model')
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto',  load_in_4bit=True,do_sample=True)
            tokenizer.pad_token = tokenizer.eos_token
        elif args.model_name == 'llama-2-13b-chat-hf':
            args.model_name = 'meta-llama/Llama-2-13b-chat-hf'
            tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = load_local_model_or_tokenizer(args.model_name, 'model')
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                             device_map='auto', load_in_4bit=True, do_sample=True)
            tokenizer.pad_token = tokenizer.eos_token
        elif args.model_name == 'Meta-Llama-3-8B':
            args.model_name = 'meta-llama/Meta-Llama-3-8B'
            tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = load_local_model_or_tokenizer(args.model_name, 'model')
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                             device_map='auto', load_in_4bit=True, do_sample=True)
            tokenizer.pad_token = tokenizer.eos_token
        elif args.model_name == 'EleutherAI/gpt-j-6B':
            tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = load_local_model_or_tokenizer(args.model_name, 'model')
            if model is None:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_q=False  )
                model = AutoModelForCausalLM.from_pretrained( "EleutherAI/gpt-j-6B", quantization_config=bnb_config)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = load_local_model_or_tokenizer(args.model_name, 'tokenizer')
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = load_local_model_or_tokenizer(args.model_name, 'model')
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(args.model_name)
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"model_name: {args.model_name}")
    return model, tokenizer


def get_label_id_dict_for_args(args: DeepArgs, tokenizer):
    if 'llama-2' in args.model_name.lower():
        label_id_dict = {k: tokenizer.encode(v, add_special_tokens=False)[1] for k, v in
                          args.label_dict.items()}
    else:
        label_id_dict = {k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in
                          args.label_dict.items()}
    for v in args.label_dict.values():
        token_num = len(tokenizer.encode(v, add_special_tokens=False))
        if token_num != 1:
            warnings.warn(f"{v} in {args.task_name} has token_num: {token_num} which is not 1")
    return label_id_dict

def get_label_id_dict_for_args_all(args: DeepArgs, tokenizer):
    if 'llama-2' in args.model_name.lower():
        label_id_dict_all = {k: tokenizer.encode(v, add_special_tokens=False)[1:] for k, v in
                          args.label_dict.items()}
    else:
        label_id_dict_all = {k: tokenizer.encode(v, add_special_tokens=False) for k, v in
                          args.label_dict.items()}
    for v in args.label_dict.values():
        token_num = len(tokenizer.encode(v, add_special_tokens=False))
        if token_num != 1:
            warnings.warn(f"{v} in {args.task_name} has token_num: {token_num} which is not 1")
    return label_id_dict_all
