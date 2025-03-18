

from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True, cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True, cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat", cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat", cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-chat", cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-chat", trust_remote_code=True, cache_dir = cache_dir)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True, cache_dir = cache_dir)