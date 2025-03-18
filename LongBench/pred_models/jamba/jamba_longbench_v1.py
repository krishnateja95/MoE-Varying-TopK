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
from transformers import AutoConfig

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="gov_report")
    return parser.parse_args(args)


def get_jamba_model(args):
    from models.Jamba.modeling_jamba import JambaForCausalLM
    model = JambaForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto", 
                                             cache_dir=cache_dir,
                                             num_experts_per_tok=args.topk)

    tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6", cache_dir=cache_dir)

    return model, tokenizer


def get_tokenized_output(prompt, tokenizer):
    
    messages = [
        {"role": "system", "content": "You are an ancient oracle who speaks in cryptic but wise phrases, always hinting at deeper meanings."},
        {"role": "user", "content": prompt},
        ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

    return inputs

def get_detokenized_output(outputs, tokenizer):
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output
    


def get_pred_jamba(args, data, max_gen, prompt_format, dataset, out_path):
    model, tokenizer = get_jamba_model(args)
    
    config = AutoConfig.from_pretrained('ai21labs/AI21-Jamba-Mini-1.6', cache_dir=cache_dir)
    
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = get_tokenized_output(prompt, tokenizer)
        
        if tokenized_prompt[0].shape[-1] + max_gen > (config.max_position_embeddings):
            print("detected large context length")
            exit()

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


# if __name__ == '__main__':
#     args = parse_args()
    
#     model2path = json.load(open("config/model2path.json", "r"))
#     model2maxlen = json.load(open("config/model2maxlen.json", "r"))
#     model_name = "AI21-Jamba-Mini-1.6"
    
#     max_length = model2maxlen[model_name]
    
#     datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
#             "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
#     dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
#     dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
#     if not os.path.exists("pred_e"):
#         os.makedirs("pred_e")
    
#     data = load_dataset('THUDM/LongBench', f"{args.dataset}_e", split='test', cache_dir=cache_dir)
#     if not os.path.exists(f"pred_e/{model_name}"):
#         os.makedirs(f"pred_e/{model_name}")
#     out_path = f"pred_e/{model_name}/{args.dataset}.jsonl"

#     if os.path.exists(out_path):
#         os.remove(out_path)

#     prompt_format = dataset2prompt[args.dataset]
#     max_gen       = dataset2maxlen[args.dataset]
    
#     get_pred(data = data,
#              max_gen = max_gen, 
#              prompt_format = prompt_format,
#              dataset = args.dataset,
#              out_path = out_path)
            
        