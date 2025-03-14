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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gov_report")
    return parser.parse_args(args)

def get_DeepSeek_V2_model():
    model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from models.DeepSeekV2.modeling_deepseekv2 import DeepseekV2ForCausalLM
    model = DeepseekV2ForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    
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

def get_pred(data, max_length, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_DeepSeek_V2_model()

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = get_DeepSeek_V2_tokenizer(prompt, tokenizer)

        if tokenized_prompt.shape[-1] + max_gen > (tokenizer.model_max_length):
            half = int(tokenizer.model_max_length/2)-(max_gen)
            first_half  = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][:half], tokenized_prompt)
            second_half = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][-half:], tokenized_prompt)
            
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
        
        pred = get_DeepSeek_V2_detokenizer(tokenizer, output[context_length:], tokenized_prompt)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    

if __name__ == '__main__':
    args = parse_args()
    
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = "DeepSeek-V2-Lite-Chat"
    
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
    
    get_pred(data = data,
             max_length = max_length,
             max_gen = max_gen, 
             prompt_format = prompt_format,
             dataset = args.dataset,
             out_path = out_path)
            
        