
import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gov_report")
    parser.add_argument('--method', type=str, default="baseline")
    parser.add_argument('--model_name', type=str, default="Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument('--topk', type=int, default="Qwen1.5-MoE-A2.7B-Chat")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("pred_e_v1"):
        os.makedirs("pred_e_v1")
    
    data = load_dataset('THUDM/LongBench', f"{args.dataset}_e", split='test', cache_dir=cache_dir)

    if not os.path.exists(f"pred_e_v1/{args.model_name}_topk_{args.topk}_method_{args.method}"):
        os.makedirs(f"pred_e_v1/{args.model_name}_topk_{args.topk}_method_{args.method}")

    out_path = f"pred_e_v1/{args.model_name}_topk_{args.topk}_method_{args.method}/{args.dataset}.jsonl"

    if os.path.exists(out_path):
        os.remove(out_path)

    prompt_format = dataset2prompt[args.dataset]
    max_gen       = dataset2maxlen[args.dataset]
    
    if args.model_name == "DeepSeek-V2-Lite-Chat":
        from pred_models.deepseek.deepseek2_longbench_v1 import get_pred_deepseekv2 as get_pred_main

    elif args.model_name == "AI21-Jamba-Mini-1.6":
        from pred_models.jamba.jamba_longbench_v1 import get_pred_jamba as get_pred_main

    elif args.model_name == "Mixtral-8x7B-Instruct-v0.1":
        from pred_models.mixtral.mixtral_longbench_v1 import get_pred_mixtral as get_pred_main

    elif args.model_name == "Phi-3.5-MoE-instruct":
        from pred_models.phi.phi3_longbench_v1 import get_pred_phi3 as get_pred_main

    elif args.model_name == "Qwen1.5-MoE-A2.7B-Chat":
        from pred_models.qwen.qwen_longbench_v1 import get_pred_qwen as get_pred_main


    get_pred_main(args = args,
             data = data,
             max_gen = max_gen, 
             prompt_format = prompt_format,
             dataset = args.dataset,
             out_path = out_path)
            
        