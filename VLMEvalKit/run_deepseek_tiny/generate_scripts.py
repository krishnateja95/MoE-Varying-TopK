

import os

content = f"""#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=24:00:00
#PBS -q by-gpu
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate Deepseek_VL

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/MoE-Varying-TopK/VLMEvalKit/

"""

run_all_file = False

for model in ["deepseek_vl2_tiny"]:
    for topk in [1, 2, 4, 6, 8, 16, 32, 64]:

        output_write_file = f"{model}_{topk}_baseline.sh"
        with open(output_write_file, "w") as file:
            file.write(content)

        op = "a" if run_all_file else "w"
        run_all_file = True
        with open("run_all_baseline_files.sh", op) as file:
            file.write(f"chmod +x {output_write_file}")
            file.write("\n")
            file.write(f"qsub {output_write_file}")
            file.write("\n")

        for dataset in [
            "AI2D_TEST_NO_MASK", 
                        "BLINK", 
                        "DocVQA_VAL", 
                        "InfoVQA_VAL", 
                        "MMMU_DEV_VAL", 
                        "RealWorldQA", 
                        "ScienceQA_VAL", 
                        "MME"
                        ]:

            work_dir = f"./all_results/baseline_model_{model}_topk_{topk}_dataset_{dataset}"
            command = f"""python run.py --topk {topk} --work-dir {work_dir} --data {dataset} --model {model} --mode all --saveresults 
"""
        
            with open(output_write_file, "a") as file:
                file.write(command)
