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
    return parser.parse_args(args)


def get_Qwen_model(args):
    model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    
    from models.Qwen2.modeling_qwen import Qwen2MoeForCausalLM
    model = Qwen2MoeForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                device_map="cuda:0",
                                                cache_dir=cache_dir,
                                                num_experts_per_tok=16
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


def get_pred(args, data, max_gen, prompt_format, dataset, out_path):
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

    

if __name__ == '__main__':
    args = parse_args()
    
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = "Qwen1.5-MoE-A2.7B-Chat"
    
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
            
        