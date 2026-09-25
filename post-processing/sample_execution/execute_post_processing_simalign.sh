#!/bin/bash
#SBATCH --job-name=PaSeMiLL_PostProcess
#SBATCH --output=slurm_postprocess_%j.out
#SBATCH --error=slurm_postprocess_%j.err
#SBATCH --time=20:00:00         
#SBATCH --gres=gpu:1             
#SBATCH --partition=lrz-hgx-a100-80x4

echo "Job started on partition: $(hostname)"
echo "start-time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SNAPP_env
echo "Conda environment 'SNAPP_env' activated."

python -u execute_post-processing_simalign.py

echo "end-time: $(date)"