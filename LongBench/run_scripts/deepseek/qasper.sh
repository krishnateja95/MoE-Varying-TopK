#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Dynamic_NAS

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/MoE-Varying-TopK/LongBench/

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 1
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 1

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 2
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 2

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 4
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 4

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 6
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 6

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 8
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 8

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 16
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 16

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 32
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 32

python3 pred_v1_main.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 64
python3 eval.py --data qasper --model DeepSeek-V2-Lite-Chat --topk 64

