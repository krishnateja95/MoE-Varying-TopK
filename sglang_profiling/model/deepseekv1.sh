



source ~/.bashrc
conda init
conda activate sglang_H100

model_dir="/vast/users/schittyvenkata/model_weights/models--deepseek-ai--deepseek-moe-16b-chat/snapshots/eefd8ac7e8dc90e095129fe1a537d5e236b2e57c"

python3 -m sglang.bench_offline_throughput --model-path $model_dir --num-prompts 10  --tp 4 --ep 4 --enable-ep-moe
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | xargs -r kill -9

# python3 -m sglang.bench_offline_throughput --model-path $model_dir --num-prompts 10  --tp 2 --ep 2

# python -m sglang.bench_one_batch --model-path $model_dir --tp 2 --ep 2 --enable-ep-moe --batch 32 --input-len 2048 --output-len 2048 --dtype float16




