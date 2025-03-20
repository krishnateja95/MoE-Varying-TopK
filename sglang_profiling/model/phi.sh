



source ~/.bashrc
conda init
conda activate sglang_H100

model_dir="/vast/users/schittyvenkata/model_weights/models--microsoft--Phi-3.5-MoE-instruct/snapshots/c5ec1449e5376ad4c7031bf0d51eabf5e7d08887"

model_name="Phi-3.5-MoE-instruct"

input_size=1024
output_size=1024
batch_size=16

for i in 1 2 4 8 16
do
    python3 update_config.py --model_dir=$model_dir --desired_topk=$i --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
    python3 bench_one_batch.py --model-path $model_dir --batch $batch_size --input-len $input_size --output-len $output_size --tp 4 --model_name=$model_name --desired_topk=$i --dtype="float16"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | xargs -r kill -9
    python3 update_config.py --model_dir=$model_dir --desired_topk=2 --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
done



for i in 1 2 4 8 16
do
    python3 update_config.py --model_dir=$model_dir --desired_topk=$i --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
    python3 bench_one_batch.py --model-path $model_dir --batch $batch_size --input-len $input_size --output-len $output_size --tp 4 --ep 4 --enable-ep-moe --model_name=$model_name --desired_topk=$i --dtype="float16"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep python | awk '{print $1}' | xargs -r kill -9
    python3 update_config.py --model_dir=$model_dir --desired_topk=2 --desired_topk_key="num_experts_per_tok" --num_layer_key="num_hidden_layers"
done




