
import torch


def split_model(model_name):
    if model_name == 'deepseek-ai/deepseek-vl2-tiny':
        return "cuda:0"
    
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
        # 'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
        'deepseek-ai/deepseek-vl2': [5,9,9,7],
    }
    num_layers_per_gpu = model_splits[model_name]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu): 
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map

model_proxy = {
    'deepseek-ai/deepseek-vl2-tiny': "deepseek_vl2_tiny",
    'deepseek-ai/deepseek-vl2-small': "deepseek_vl2_small",
    'deepseek-ai/deepseek-vl2': "deepseek_vl2"
            }

from .deepseek_model.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
model = DeepseekVLV2ForCausalLM.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                torch_dtype=torch.bfloat16,
                                                device_map = self.device_map,
                                                num_experts_per_tok = args.topk,
                                                cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
