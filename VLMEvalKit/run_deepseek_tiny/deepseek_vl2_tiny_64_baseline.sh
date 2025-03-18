#!/bin/bash -l
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

python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_AI2D_TEST_NO_MASK --data AI2D_TEST_NO_MASK --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_BLINK --data BLINK --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_DocVQA_VAL --data DocVQA_VAL --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_InfoVQA_VAL --data InfoVQA_VAL --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_MMMU_DEV_VAL --data MMMU_DEV_VAL --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_RealWorldQA --data RealWorldQA --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_ScienceQA_VAL --data ScienceQA_VAL --model deepseek_vl2_tiny --mode all --saveresults 
python run.py --topk 64 --work-dir ./all_results/baseline_model_deepseek_vl2_tiny_topk_64_dataset_MME --data MME --model deepseek_vl2_tiny --mode all --saveresults 
