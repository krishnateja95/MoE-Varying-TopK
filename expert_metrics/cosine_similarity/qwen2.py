import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse


if __name__ == '__main__':

    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

    model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                cache_dir=cache_dir,
                                                )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    print(model)