import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'


def get_mixtral_model(args):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    
    from models.Mixtral8x7b.modeling_mixtral import MixtralForCausalLM
    model = MixtralForCausalLM.from_pretrained(model_name, 
                                                 cache_dir = cache_dir, 
                                                 device_map = "auto", 
                                                 torch_dtype = torch.float16,
                                                 _attn_implementation = "flash_attention_2",
                                                 num_experts_per_tok=args.topk
                                                 )
    return model, tokenizer


def get_tokenized_output(prompt, tokenizer):
    messages = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    return inputs


def get_detokenized_output(outputs, tokenizer):
    return tokenizer.decode(outputs, skip_special_tokens=True)
    

def get_pred_mixtral(args, data, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_mixtral_model(args)
    
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = get_tokenized_output(prompt, tokenizer)
        
        if tokenized_prompt[0].shape[-1] + max_gen > (tokenizer.model_max_length):
            half = int(tokenizer.model_max_length/2)-(max_gen)
            first_half  = get_detokenized_output(tokenizer, tokenized_prompt[0][:half])
            second_half = get_detokenized_output(tokenizer, tokenized_prompt[0][-half:])
            prompt =  first_half + second_half
            tokenized_prompt = get_tokenized_output(prompt, tokenizer)

        tokenized_prompt = tokenized_prompt.to(device = "cuda:0")
        
        context_length = tokenized_prompt[0].shape[-1]

        if dataset == "samsum":
            output = model.generate(
                tokenized_prompt,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                tokenized_prompt,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

