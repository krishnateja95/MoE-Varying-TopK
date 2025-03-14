#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=20:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Dynamic_NAS

# "narrativeqa" "qasper": 128,
# "multifieldqa_en": 64,
# "multifieldqa_zh": 64,
# "hotpotqa": 32,
# "2wikimqa": 32,
# "musique": 32,
# "dureader": 128,
# "gov_report": 512,
# "qmsum": 512,
# "multi_news": 512,
# "vcsum": 512,
# "trec": 64,
# "triviaqa": 32,
# "samsum": 128,
# "lsht": 64,
# "passage_count": 32,
# "passage_retrieval_en": 32,
# "passage_retrieval_zh": 32,
# "lcc": 64,
# "repobench-p": 64

# datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
#         "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

python3 deepseek_v2.py --dataset="multi_news"



