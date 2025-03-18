import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer, GenerationConfig
import torch.multiprocessing as mp
import torch

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

template_rag           = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context    = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot         = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot     = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

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

def query_llm(prompt, model, tokenizer, temperature=0.5, max_new_tokens=128, stop=None):
    
    tokenized_prompt = get_DeepSeek_V2_tokenizer(prompt, tokenizer)
    
    if tokenized_prompt.shape[-1] + max_new_tokens > (tokenizer.model_max_length):
        half = int(tokenizer.model_max_length/2)-(max_new_tokens)
        first_half  = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][:half], tokenized_prompt)
        second_half = get_DeepSeek_V2_detokenizer(tokenizer, tokenized_prompt[0][-half:], tokenized_prompt)
        
        prompt =  first_half + second_half
        tokenized_prompt = get_DeepSeek_V2_tokenizer(prompt, tokenizer)

    
    tokenized_prompt = tokenized_prompt.to(device = "cuda:0")
    context_length = tokenized_prompt.shape[-1]

    output = model.generate(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )[0]

    pred = get_DeepSeek_V2_detokenizer(tokenizer, output[context_length:], tokenized_prompt)

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
    
    model, tokenizer = get_DeepSeek_V2_model()
    
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

def main_phi3(args):
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

    if os.path.exists(out_file):
        os.remove(out_file)

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






#     dataset = load_dataset('THUDM/LongBench-v2', split='train', cache_dir=cache_dir)
#     data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

#     has_data = {}
#     if os.path.exists(out_file):
#         with open(out_file, encoding='utf-8') as f:
#             has_data = {json.loads(line)["_id"]: 0 for line in f}
#     fout = open(out_file, 'a', encoding='utf-8')
#     data = []
#     for item in data_all:
#         if item["_id"] not in has_data:
#             data.append(item)

#     data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
#     processes = []
#     for rank in range(args.n_proc):
#         p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()

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
#     main_phi3(args)