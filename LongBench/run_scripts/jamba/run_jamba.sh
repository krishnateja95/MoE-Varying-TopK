#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=20:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Dynamic_NAS


python3 run_jamba.py --dataset="gov_report"




