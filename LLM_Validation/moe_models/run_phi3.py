import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import pipeline

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gov_report")
    return parser.parse_args(args)


def get_phi_model(args):
    model_name = "microsoft/Phi-3.5-MoE-instruct"
    from models.Phi3.modeling_phi import PhiMoEForCausalLM

    model = PhiMoEForCausalLM.from_pretrained( 
        model_name,  
        device_map="auto",  
        torch_dtype=torch.float16,  
        trust_remote_code=True, 
        cache_dir = cache_dir 
    ) 

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir) 
    
    return model, tokenizer

def get_pred(args, data, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_phi_model(args)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer
                    ) 


    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}, 
            {"role": "user", "content": prompt}
            ] 

        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        context_length = tokenized_prompt.input_ids[0].shape[-1]

        if dataset == "samsum":
            generation_args = {
                "max_new_tokens": max_gen, 
                "return_full_text": False,
                "num_beams":1, 
                "temperature": 1.0, 
                "do_sample": False,
                "min_length":context_length+1,
                "eos_token_id": [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]] 
            } 
        else:
            generation_args = {
                "return_full_text": False, 
                "temperature": 1.0, 
                "do_sample": False,
                "max_new_tokens": max_gen,
                "num_beams":1,
            }


        pred = pipe(messages, **generation_args) 
        print(pred)
        print(pred[0]['generated_text'])

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    

if __name__ == '__main__':
    args = parse_args()
    
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = "Phi-3.5-MoE-instruct"
    
    max_length = model2maxlen[model_name]
    
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    
    data = load_dataset('THUDM/LongBench', f"{args.dataset}_e", split='test', cache_dir=cache_dir)
    if not os.path.exists(f"pred_e/{model_name}"):
        os.makedirs(f"pred_e/{model_name}")
    out_path = f"pred_e/{model_name}/{args.dataset}.jsonl"

    if os.path.exists(out_path):
        os.remove(out_path)

    prompt_format = dataset2prompt[args.dataset]
    max_gen       = dataset2maxlen[args.dataset]
    
    get_pred(args = args, 
             data = data,
             max_gen = max_gen, 
             prompt_format = prompt_format,
             dataset = args.dataset,
             out_path = out_path)
            
        