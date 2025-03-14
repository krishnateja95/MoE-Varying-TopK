import sys
import torch
from transformers import AutoModelForCausalLM
import warnings
from .base import BaseModel
from ..smp import *
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model

def quant_split_model(model_name):
    if model_name == 'deepseek-ai/deepseek-vl2-tiny':
        return "cuda:0"
    
    device_map = {}
    model_splits = {
        'deepseek-ai/deepseek-vl2-small': [13, 14], # 2 GPU for 16b
        'deepseek-ai/deepseek-vl2': [10, 10, 10], # 3 GPU for 27b
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

class DeepSeekVL2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='deepseek-ai/deepseek-vl2-tiny', args=None, **kwargs):
        
        assert model_path is not None
        self.model_path = model_path
        self.device_map = split_model(model_path)

        if args.model_precision == "uniform_quant":

            if args.quant_format == "hqq":
                from .deepseek_model.processing_deepseek_vl_v2 import DeepseekVLV2Processor
                from .DeepSeek_VL2_HQQ.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

                self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
                self.tokenizer = self.vl_chat_processor.tokenizer

                self.model = DeepseekVLV2ForCausalLM.from_pretrained(model_path,
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                device_map = self.device_map,
                                                                cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
                
                self.model.quantize_experts(nbits = args.bits, group_size = 64, quant_config_dict = None)

                # model_proxy = {
                #     'deepseek-ai/deepseek-vl2-tiny': "deepseek_vl2_tiny",
                #     'deepseek-ai/deepseek-vl2-small': "deepseek_vl2_small",
                #     'deepseek-ai/deepseek-vl2': "deepseek_vl2"
                # }
                
                # self.model, self.vl_chat_processor, self.tokenizer = quantized_model(bits = args.bits,
                #                                                                     model_name=model_path,
                #                                                                     format=args.quant_format,
                #                                                                     device_map=self.device_map,
                #                                                                     )

                # print(self.model)
                # exit()


            else:

                torch.set_grad_enabled(True)
                torch.enable_grad()

                model_proxy = {
                    'deepseek-ai/deepseek-vl2-tiny': "deepseek_vl2_tiny",
                    'deepseek-ai/deepseek-vl2-small': "deepseek_vl2_small",
                    'deepseek-ai/deepseek-vl2': "deepseek_vl2"
                }
                
                from .deepseek_vl2_quant import quantized_model
                self.model, self.vl_chat_processor, self.tokenizer = quantized_model(bits = args.bits,
                                                                                    model_name=model_path,
                                                                                    format=args.quant_format,
                                                                                    device_map=self.device_map,
                                                                                    )

                
                print(self.model)
                torch.set_grad_enabled(False)

            if "tiny" in model_path:
                self.model = self.model.cuda()
            
            else:
                self.model = dispatch_model(self.model, device_map=self.device_map)
                
            if args.quant_format == "hqq":
                self.model = self.model.to(torch.bfloat16).eval()
            else:
                self.model = self.model.to(torch.bfloat16).eval()

        elif args.model_precision == "mixed_precision":
             
            torch.set_grad_enabled(True)
            torch.enable_grad()

            model_proxy = {
                'deepseek-ai/deepseek-vl2-tiny': "deepseek_vl2_tiny",
                'deepseek-ai/deepseek-vl2-small': "deepseek_vl2_small",
                'deepseek-ai/deepseek-vl2': "deepseek_vl2"
            }

            if args.assignment == "per_layer":
                dir = "/lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/github_MoE_Mixed_precision_NAS/MoE-Mixed-Prec/VLMEvalKit/vlmeval/vlm/DeepSeek_VL2/json_files/"
                json_file = dir  + f"{model_proxy[model_path]}_{args.assignment}_importance_{args.importance}_n_clusters_{args.n_clusters}_mixed_precision_quant.json"


            elif args.assignment == "per_model":
                dir = "/lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/github_MoE_Mixed_precision_NAS/MoE-Mixed-Prec/VLMEvalKit/vlmeval/vlm/DeepSeek_VL2/json_files/"
                json_file = dir  + f"{model_proxy[model_path]}_per_model_{args.importance}_n_clusters_{args.n_clusters}_mixed_precision_quant.json"

            from .deepseek_vl2_quant import quantized_model
            self.model, self.vl_chat_processor, self.tokenizer = quantized_model(bits = args.bits,
                                                                                 model_name=model_path,
                                                                                 format=args.quant_format,
                                                                                 device_map=self.device_map,
                                                                                 json_file = json_file, 
                                                                                 )

            
            print(self.model)
            torch.set_grad_enabled(False)
            
            if "tiny" in model_path:
                self.model = self.model.cuda()
            
            else:
                self.model = dispatch_model(self.model, device_map=self.device_map)
            
            if args.quant_format == "hqq":
                self.model = self.model.to(torch.bfloat16).eval()
            else:
                self.model = self.model.to(torch.bfloat16).eval()


        elif args.model_precision == "activation_frequency_profiling":
            from .DeepSeek_VL2_quant.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
            from .DeepSeek_VL2_quant.processing_deepseek_vl_v2 import DeepseekVLV2Processor

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model = DeepseekVLV2ForCausalLM.from_pretrained(model_path,
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                device_map = self.device_map,
                                                                cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
            
        elif args.model_precision == "fp_baseline":
            from .deepseek_model.processing_deepseek_vl_v2 import DeepseekVLV2Processor
            from .deepseek_model.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

            self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
            self.tokenizer = self.vl_chat_processor.tokenizer

            self.model = DeepseekVLV2ForCausalLM.from_pretrained(model_path,
                                                            trust_remote_code=True,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map = self.device_map,
                                                            cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/')
            # print(model.hf_device_map)
            # print(model.image_newline.device)
            # print(model.device)
            # exit()
            # self.model = model.cuda().eval()
        
        self.model = self.model.eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')


    def prepare_inputs(self, message, dataset=None):

        if dataset == 'MMMU_DEV_VAL':

            def prepare_itlist(msgs):
                content, images = '', []
                image_idx = 1
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += f'<image {image_idx}>'
                        image_idx += 1
                    elif s['type'] == 'text':
                        content += s['value']
                # content = '<image>' * (image_idx-1) + '\n' + content
                content = '<image>' * (image_idx - 1) + '\n' + content
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                content = content.replace(
                    'Please select the correct answer from the options above.',
                    "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                )
                content = content.replace('Question:', "")
                content = content.replace('Options:\n', "")
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    content = content.replace(
                        'Please select the correct answer from the options above.',
                        "Answer with the option's letter from the given choices directly. Answer the question using a single word or phrase.\n"  # noqa
                    )
                    content = content.replace('Question:', "")
                    content = content.replace('Options:\n', "")
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        else:

            def prepare_itlist(msgs):
                content, images = '', []
                for s in msgs:
                    if s['type'] == 'image':
                        images.append(s['value'])
                        content += '<image>\n'
                    elif s['type'] == 'text':
                        content += s['value']
                return content, images

            conversation = []
            if 'role' not in message[0]:
                content, images = prepare_itlist(message)
                conversation.append(dict(role='<|User|>', content=content, images=images))
            else:
                role_map = {'user': '<|User|>', 'assistant': '<|Assistant|>'}
                for msgs in message:
                    role = role_map[msgs['role']]
                    content, images = prepare_itlist(msgs['content'])
                    conversation.append(dict(role=role, content=content, images=images))
            conversation.append(dict(role='<|Assistant|>', content=''))

        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message, dataset)
        from .deepseek_model.io_utils import load_pil_images
        # from deepseek_vl2.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)

        if dataset == 'MMMU_DEV_VAL':
            if len(pil_images):
                h, w = pil_images[0].size
                pil_images[0] = pil_images[0].resize((2 * h, 2 * w), Image.BILINEAR)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )

        # prepare_inputs = prepare_inputs.to(self.model.device)
        # prepare_inputs = prepare_inputs.to(self.model.image_newline.device)
        prepare_inputs = prepare_inputs.to("cuda:0")
        
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512
        )

        # run the model to get the response
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=True
        )
        answer = answer.rstrip('.')

        return answer

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset=dataset)
