



source ~/.bashrc
conda init
conda activate sglang_H100

model_dir="/vast/users/schittyvenkata/model_weights/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"

model_name="DeepSeek-V2-Lite-Chat"

input_size=1024
output_size=1024
batch_size=16

# for i in 1 2 4 6 8 16 32 64
# do
#     python3 update_config.py --model_dir=$model_dir --desired_topk=$i --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
#     python3 bench_one_batch.py --model-path $model_dir --batch $batch_size --input-len $input_size --output-len $output_size --tp 4 --model_name=$model_name --desired_topk=$i --dtype="float16"
#     nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | xargs -r kill -9
#     python3 update_config.py --model_dir=$model_dir --desired_topk=2 --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
# done



for i in 1 2 4 6 8 16 32 64
do
    python3 update_config.py --model_dir=$model_dir --desired_topk=$i --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
    python3 bench_one_batch.py --model-path $model_dir --batch $batch_size --input-len $input_size --output-len $output_size --tp 4 --ep 4 --enable-ep-moe --model_name=$model_name --desired_topk=$i --dtype="float16"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | xargs -r kill -9
    python3 update_config.py --model_dir=$model_dir --desired_topk=2 --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
done




