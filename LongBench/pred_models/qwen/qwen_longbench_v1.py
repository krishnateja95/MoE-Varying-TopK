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

def get_Qwen_model(args):
    model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    
    from models.Qwen2.modeling_qwen import Qwen2MoeForCausalLM
    model = Qwen2MoeForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                device_map="cuda:0",
                                                cache_dir=cache_dir,
                                                num_experts_per_tok=args.topk
                                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    return model, tokenizer


def get_qwen_tokenizer(prompt, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    return model_inputs


def get_qwen_detokenizer(tokenizer, outputs):
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    return response


def get_pred_qwen(args, data, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_Qwen_model(args)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = get_qwen_tokenizer(prompt, tokenizer)
        
        if tokenized_prompt.input_ids[0].shape[-1] + max_gen > (tokenizer.model_max_length):
            half = int(tokenizer.model_max_length/2)-(max_gen)
            first_half  = get_qwen_detokenizer(tokenizer, tokenized_prompt.input_ids[0][:half])
            second_half = get_qwen_detokenizer(tokenizer, tokenized_prompt.input_ids[0][-half:])
            
            prompt =  first_half + second_half
            tokenized_prompt = get_qwen_tokenizer(prompt, tokenizer)

        tokenized_prompt = tokenized_prompt.to(device = "cuda:0")
        
        context_length = tokenized_prompt.input_ids[0].shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **tokenized_prompt,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **tokenized_prompt,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        
        pred = get_qwen_detokenizer(tokenizer, output[context_length:])
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    