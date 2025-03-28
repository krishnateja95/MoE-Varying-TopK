#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=72:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles
module load conda
conda activate TRT_LLM_TopK

model_dir="/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
trt_checkpoint="/lus/grand/projects/datascience/krishnat/home_dir_code/trtllm_model_paths/trt_checkpoints/deepseek_v2_moe/"
trt_engine="/lus/grand/projects/datascience/krishnat/home_dir_code/trtllm_model_paths/trt_engine/deepseek_v2_moe"

export LD_LIBRARY_PATH="/lus/eagle/projects/datascience/krishnat95/envs/TRT_LLM_TopK/lib"

model_name="deepseek-ai/DeepSeek-V2-Lite-Chat"

TP_size=4
MoE_TP_size=4
MoE_EP_size=1
for precision in "float16"; do
    # rm -rf $trt_checkpoint/*
    # rm -rf $trt_engine/*
    python convert_checkpoint.py --workers=4 --tp_size=$TP_size --moe_tp_size=$MoE_TP_size --moe_ep_size=$MoE_EP_size --model_dir=$model_dir --output_dir=$trt_checkpoint --dtype=$precision
    for batch_size in 128; do
        for input_output_length in 4096; do
            trtllm-build --workers=4 --checkpoint_dir=$trt_checkpoint --output_dir=$trt_engine --gemm_plugin=$precision --gpt_attention_plugin=$precision --max_batch_size=$batch_size --max_input_len=$input_output_length
            # mpirun --oversubscribe -np $TP_size python3 ../run.py --model_name=$model_name --tokenizer_dir=$model_dir --engine_dir=$trt_engine --max_output_len=$input_output_length --max_input_length=$input_output_length --run_profiling --batch_size=$batch_size 
        done
    done
done


# TP_size=4
# EP_size=1
# MoE_TP_size=1
# MoE_EP_size=1
# for precision in "float16"; do
#     rm -rf $trt_checkpoint/*
#     rm -rf $trt_engine/*
#     python convert_checkpoint.py --workers=4 --tp_size=$TP_size --ep_size=$EP_size --moe_tp_size=$MoE_TP_size --moe_ep_size=$MoE_EP_size --model_dir=$model_dir --output_dir=$trt_checkpoint --dtype=$precision
#     for batch_size in 128; do
#         for input_output_length in 4096; do
#             trtllm-build --workers=4 --checkpoint_dir=$trt_checkpoint --output_dir=$trt_engine --gemm_plugin=$precision --gpt_attention_plugin=$precision --max_batch_size=$batch_size --max_input_len=$input_output_length --max_output_len=$input_output_length
#             mpirun -np $tensor_parallel python3 ../run.py --model_name=$model_name --tokenizer_dir=$model_dir --engine_dir=$trt_engine --max_output_len=$input_output_length --max_input_length=$input_output_length --run_profiling --batch_size=$batch_size 
#         done
#     done
# done





# TP_size=4
# EP_size=1
# MoE_TP_size=1
# MoE_EP_size=1
# for precision in "float16"; do
#     rm -rf $trt_checkpoint/*
#     rm -rf $trt_engine/*
#     python convert_checkpoint.py --workers=4 --tp_size=$TP_size --ep_size=$EP_size --moe_tp_size=$MoE_TP_size --moe_ep_size=$MoE_EP_size --model_dir=$model_dir --output_dir=$trt_checkpoint --dtype=$precision
#     for batch_size in 128; do
#         for input_output_length in 4096; do
#             trtllm-build --workers=4 --checkpoint_dir=$trt_checkpoint --output_dir=$trt_engine --gemm_plugin=$precision --gpt_attention_plugin=$precision --max_batch_size=$batch_size --max_input_len=$input_output_length --max_output_len=$input_output_length
#             mpirun -np $tensor_parallel python3 ../run.py --model_name=$model_name --tokenizer_dir=$model_dir --engine_dir=$trt_engine --max_output_len=$input_output_length --max_input_length=$input_output_length --run_profiling --batch_size=$batch_size 
#         done
#     done
# done
