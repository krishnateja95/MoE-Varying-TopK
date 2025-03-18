

import os

content = f"""#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Dynamic_NAS

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/MoE-Varying-TopK/LongBench/

"""
model_name = "Phi-3.5-MoE-instruct"
run_all_file = False

for dataset in [
                "qasper",
                "multifieldqa_en",
                "hotpotqa",
                # "2wikimqa",
                "gov_report",
                # "multi_news",
                # "trec",
                "triviaqa",
                # "samsum",
                # "passage_count",
                # "passage_retrieval_en",
                # "lcc",
                # "repobench-p",
                # "narrativeqa",
                "qmsum",
                # "musique"
                ]:

    
    output_write_file = f"{dataset}.sh"

    with open(output_write_file, "w") as file:
        file.write(content)

    op = "a" if run_all_file else "w"
    run_all_file = True
    with open("run_all_datasets.sh", op) as file:
        file.write(f"chmod +x {output_write_file}")
        file.write("\n")
        file.write(f"qsub {output_write_file}")
        file.write("\n")

        
    for topk in [1, 2, 4, 8, 16]:
        command = f"""python3 pred_v1_main.py --data {dataset} --model {model_name} --topk {topk}
python3 eval.py --data {dataset} --model {model_name} --topk {topk}

"""
        with open(output_write_file, "a") as file:
            file.write(command)


