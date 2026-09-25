#!/bin/bash
#SBATCH --job-name=nllb_eval3
#SBATCH --output=eval3_%j.log
#SBATCH --error=eval3_%j.err
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G

echo "Job started on node: $(hostname)"
echo "start-time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SNAPP_env
echo "Conda environment 'SNAPP_env' activated."

python ../scripts/translate.py

echo "end-time: $(date)"
