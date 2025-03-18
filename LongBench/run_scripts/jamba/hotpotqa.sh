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

python3 pred_v1_main.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 1
python3 eval.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 1

python3 pred_v1_main.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 2
python3 eval.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 2

python3 pred_v1_main.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 4
python3 eval.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 4

python3 pred_v1_main.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 8
python3 eval.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 8

python3 pred_v1_main.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 16
python3 eval.py --data hotpotqa --model AI21-Jamba-Mini-1.6 --topk 16

