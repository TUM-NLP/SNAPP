#!/bin/bash
#SBATCH --job-name=nllb_finetune
#SBATCH --output=finetune_%j.log
#SBATCH --error=finetune_%j.err
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

echo "Job started on node: $(hostname)"
echo "start-time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SNAPP_env 
echo "Conda environment 'SNAPP_env' activated."

python ../scripts/fine_tune_nllb_reverse.py

echo "end-time: $(date)"
