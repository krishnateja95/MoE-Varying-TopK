from vlmeval.vlm import *
from functools import partial

molmo_series={
    'molmoE-1B-0924': partial(molmo, model_path='allenai/MolmoE-1B-0924'),
    # 'molmo-7B-D-0924': partial(molmo, model_path='allenai/Molmo-7B-D-0924'),
    # 'molmo-7B-O-0924': partial(molmo, model_path='allenai/Molmo-7B-O-0924'),
}

llama_series={
    'Llama-3.2-11B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct'),
}

qwen2vl_series = {
    'Qwen2-VL-2B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2.5-VL-3B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-3B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28, use_custom_prompt=False),
    'Qwen2.5-VL-7B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-7B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28, use_custom_prompt=False),
}



phi3_series = {
    'Phi-3.5-Vision': partial(Phi3_5Vision, model_path='microsoft/Phi-3.5-vision-instruct')
}

supported_VLM = {}

model_groups = [llama_series, molmo_series, qwen2vl_series, phi3_series]

for grp in model_groups:
    supported_VLM.update(grp)
