
import os

base_content = f"""#!/bin/bash -l
#COBALT -t 6:00:00 -n 1 -q gpu_h100

source ~/.bashrc
conda init
conda activate sglang_H100

"""

run_all_file = False

for model in ["molmoE-1B-0924"]:

    for dataset in [
                    "AI2D_TEST_NO_MASK", 
            "MME", 
            "MMMU_DEV_VAL",
            "DocVQA_VAL", 
            "InfoVQA_VAL", 
            "RealWorldQA", 
            "ScienceQA_VAL", 
            "BLINK"
                    ]:
        
        output_write_file = f"{model}_{dataset}_run_profile_fp16.sh"

        with open(output_write_file, "w") as file:
            file.write(base_content)

        #baseline
        model_precision = "fp_baseline"
        work_dir = f"./all_results_no_gate/{model}_bits_16_{model_precision}_{dataset}"
                    
        command = f"""python run.py --bits 16 --model_precision {model_precision} --work-dir {work_dir} --data {dataset} --model {model} --mode all --saveresults 
"""
            
        with open(output_write_file, "a") as file:
            file.write(command)
        
        op = "a" if run_all_file else "w"
        run_all_file = True
        with open("run_all_fp16.sh", op) as file:
            file.write(f"chmod +x {output_write_file}")
            file.write("\n")
            file.write(f"qsub {output_write_file}")
            file.write("\n")

        
#         for bits in [8, 4]:
            
#             output_write_file = f"{model}_{dataset}_{bits}.sh"
#             with open(output_write_file, "w") as file:
#                 file.write(base_content)

#             for quant_format in ["auto_gptq", "auto_awq"]:
                
#                 model_precision = "uniform_quant"
#                 work_dir = f"./all_results_no_gate/{model}_bits_{bits}_{model_precision}_{model_precision}_{dataset}_{quant_format}"
#                 command = f"""python run.py --bits {bits} --model_precision {model_precision} --quant_format {quant_format} --work-dir {work_dir} --data {dataset} --model {model} --mode all --saveresults 
# """
            
#                 with open(output_write_file, "a") as file:
#                     file.write(command)

#                 with open("run_all_baseline_files.sh", "a") as file:
#                     file.write(f"chmod +x {output_write_file}")
#                     file.write("\n")
#                     file.write(f"qsub {output_write_file}")
#                     file.write("\n")


                
