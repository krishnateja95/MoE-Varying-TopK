import os
from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             cache_dir=cache_dir,
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             _attn_implementation = "flash_attention_2",
                                             device_map = "auto")


def get_model_cosine_similarity(model, layer_id):

    cosine_similarity_dict = {}
    
    for expert_index_i in range(64):
        layer_dict = {}

        for expert_index_j in range(64):
            print(layer_id, expert_index_i, expert_index_j)
                
            for name, param in model.named_parameters():

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_i}.gate_proj.weight":
                    expert_index_i_gate_proj = param

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_j}.gate_proj.weight":
                    expert_index_j_gate_proj = param

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_i}.up_proj.weight":
                    expert_index_i_up_proj = param

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_j}.up_proj.weight":
                    expert_index_j_up_proj = param

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_i}.down_proj.weight":
                    expert_index_i_down_proj = param

                if name == f"model.layers.{layer_id}.mlp.experts.{expert_index_j}.down_proj.weight":
                    expert_index_j_down_proj = param

            layer_dict[expert_index_j] = cosim(expert_index_i_gate_proj, expert_index_j_gate_proj) + cosim(expert_index_i_up_proj, expert_index_j_up_proj) + cosim(expert_index_i_down_proj, expert_index_j_down_proj) 
    
        cosine_similarity_dict[expert_index_i] = layer_dict
            
    return cosine_similarity_dict



def cosim(expert1, expert2, eps: float = 1e-8):
    v1 = expert1.flatten()
    v2 = expert2.flatten()
    
    if v1.shape != v2.shape:
        raise ValueError(f"Expert tensors must have same number of elements. Got {v1.numel()} vs {v2.numel()}")
    
    # Compute similarity components
    dot_product = torch.dot(v1, v2)
    norm1 = torch.norm(v1, p=2)
    norm2 = torch.norm(v2, p=2)
    
    # Handle zero-norm cases
    if norm1 < eps or norm2 < eps:
        return torch.tensor(0.0, device=expert1.device)  # Consider zero vectors as orthogonal
    
    return float(dot_product / (norm1 * norm2 + eps)) 


layer_id = 1
cosine_similarity = get_model_cosine_similarity(model, layer_id)
print(cosine_similarity)

from plot_heatmap import plot_metric_heatmap_deepseek
plot_metric_heatmap_deepseek(cosine_similarity,
                             fig_title = f"Cosine Similarity Map between Expert of Layer {layer_id}", 
                             heatmap_title = " ",
                             filename = f"Plots/DeepseekV2/Cosine_similarity_layer_{layer_id}",
                             metric_type = "Cosine Similarity")


