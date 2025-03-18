import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer, GenerationConfig, AutoConfig
import torch.multiprocessing as mp
import torch
from accelerate import infer_auto_device_map

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

template_rag           = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context    = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot         = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot     = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()


def get_jamba_model(args=None):
    from models.Jamba.modeling_jamba import JambaForCausalLM

    # device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1,
    #               'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 
    #               'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 2, 'model.layers.16': 2,
    #               'model.layers.17': 2, 'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2,
    #               'model.layers.23': 3, 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3, 'model.layers.28': 3,
    #               'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.final_layernorm': 3, 'lm_head': 3}

    device_map = "auto"
    model = JambaForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map=device_map, 
                                             cache_dir=cache_dir)

    # max_memory = {0: "30GiB", 1: "30GiB", 2: "30GiB", 3: "30GiB"}
    # device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"])
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
    

def query_llm(prompt,
              model,
              tokenizer, 
              temperature=0.5,
              max_new_tokens=128
              ):
    
    config = AutoConfig.from_pretrained('ai21labs/AI21-Jamba-Mini-1.6', cache_dir=cache_dir)
    tokenized_prompt = get_tokenized_output(prompt, tokenizer)
    
    if tokenized_prompt[0].shape[-1] + max_new_tokens > (config.max_position_embeddings):
        half = int(config.max_position_embeddings/2)-(max_new_tokens)

        first_half  = tokenizer.decode(tokenized_prompt[0][:half], skip_special_tokens=True)
        second_half = tokenizer.decode(tokenized_prompt[0][-half:], skip_special_tokens=True)
        prompt =  first_half + second_half
        
        tokenized_prompt = get_tokenized_output(prompt, tokenizer)
        
    tokenized_prompt = tokenized_prompt.to(device = "cuda:0")
    context_length = tokenized_prompt.shape[-1]

    output = model.generate(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )[0]

    # pred = get_detokenized_output(tokenizer, output[context_length:])
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    
    print(pred)

    exit()
    return pred

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):
    
    model, tokenizer = get_jamba_model()
    
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model, tokenizer, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, model, tokenizer, temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot:
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main_jamba(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train', cache_dir=cache_dir)
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    get_pred(data, args, fout)

    
    # processes = []
    # for rank in range(args.n_proc):
    #     p = mp.Process(target=func(), args=(data_subsets[rank], args, fout))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--save_dir", "-s", type=str, default="results")
#     parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
#     parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
#     parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
#     parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
#     parser.add_argument("--n_proc", "-n", type=int, default=4)
#     parser.add_argument("--topk", type=int, default=4)

#     args = parser.parse_args()
#     main_jamba(args)