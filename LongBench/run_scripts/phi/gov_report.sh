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

python3 pred_v1_main.py --data gov_report --model Phi-3.5-MoE-instruct --topk 1
python3 eval.py --data gov_report --model Phi-3.5-MoE-instruct --topk 1

python3 pred_v1_main.py --data gov_report --model Phi-3.5-MoE-instruct --topk 2
python3 eval.py --data gov_report --model Phi-3.5-MoE-instruct --topk 2

python3 pred_v1_main.py --data gov_report --model Phi-3.5-MoE-instruct --topk 4
python3 eval.py --data gov_report --model Phi-3.5-MoE-instruct --topk 4

python3 pred_v1_main.py --data gov_report --model Phi-3.5-MoE-instruct --topk 8
python3 eval.py --data gov_report --model Phi-3.5-MoE-instruct --topk 8

python3 pred_v1_main.py --data gov_report --model Phi-3.5-MoE-instruct --topk 16
python3 eval.py --data gov_report --model Phi-3.5-MoE-instruct --topk 16

