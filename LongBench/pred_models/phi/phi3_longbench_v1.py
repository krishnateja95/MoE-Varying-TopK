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
        cache_dir = cache_dir,
        num_experts_per_tok = args.topk 
    ) 

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir) 
    
    return model, tokenizer

def get_pred_phi3(args, data, max_gen, prompt_format, dataset, out_path):
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
            json.dump({"pred": pred[0]['generated_text'], "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    
