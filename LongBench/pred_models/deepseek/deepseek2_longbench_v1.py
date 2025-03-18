import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def get_DeepSeek_V2_model(args):
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from models.DeepSeekV2.modeling_deepseekv2 import DeepseekV2ForCausalLM
    model = DeepseekV2ForCausalLM.from_pretrained(model_name,
                                                  cache_dir=cache_dir,
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16,
                                                  num_experts_per_tok = args.topk,
                                                  _attn_implementation = "flash_attention_2",
                                                  device_map = "auto")

    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer

def get_DeepSeek_V2_output(input_tensor, model, tokenizer, max_gen):
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=max_gen)
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return result

def get_DeepSeek_V2_tokenizer(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return input_tensor

def get_DeepSeek_V2_detokenizer(tokenizer, outputs):
    result = tokenizer.decode(outputs, skip_special_tokens=True)
    return result

def get_pred_deepseekv2(args, data, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_DeepSeek_V2_model(args)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = get_DeepSeek_V2_tokenizer(prompt, tokenizer)

        if tokenized_prompt.shape[-1] + max_gen > (tokenizer.model_max_length):
            half = int(tokenizer.model_max_length/2)-(max_gen)
            first_half  = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][:half])
            second_half = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][-half:])
            
            prompt =  first_half + second_half
            tokenized_prompt = get_DeepSeek_V2_tokenizer(prompt, tokenizer)

        tokenized_prompt = tokenized_prompt.to(device = "cuda:0")
        context_length = tokenized_prompt.shape[-1]

        if tokenized_prompt.shape[-1] + max_gen >= tokenizer.model_max_length:
            print("context_length", context_length, tokenized_prompt.shape[-1], tokenizer.model_max_length, half)

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
        
        pred = get_DeepSeek_V2_detokenizer(tokenizer, output[context_length:])

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')





