#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=24:00:00
#PBS -q by-gpu
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate MoE_Mixed_Precision

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/MoE-Varying-TopK/VLMEvalKit/

python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_AI2D_TEST_NO_MASK --data AI2D_TEST_NO_MASK --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_BLINK --data BLINK --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_DocVQA_VAL --data DocVQA_VAL --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_InfoVQA_VAL --data InfoVQA_VAL --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_MMMU_DEV_VAL --data MMMU_DEV_VAL --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_RealWorldQA --data RealWorldQA --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_ScienceQA_VAL --data ScienceQA_VAL --model molmoE-1B-0924 --mode all --saveresults 
python run.py --topk 4 --work-dir ./all_results/baseline_model_molmoE-1B-0924_topk_4_dataset_MME --data MME --model molmoE-1B-0924 --mode all --saveresults 
