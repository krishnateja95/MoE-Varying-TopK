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

python3 pred_v1_main.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 1
python3 eval.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 1

python3 pred_v1_main.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 2
python3 eval.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 2

python3 pred_v1_main.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 4
python3 eval.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 4

python3 pred_v1_main.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 8
python3 eval.py --data multifieldqa_en --model Mixtral-8x7B-Instruct-v0.1 --topk 8

