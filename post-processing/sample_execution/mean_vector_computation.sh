#!/bin/bash
#SBATCH --job-name=MeanVecCalc
#SBATCH --output=slurm_meanvec_%j.out
#SBATCH --error=slurm_meanvec_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=lrz-hgx-a100-80x4

echo "Job started on partition: $(hostname)"
echo "start-time: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SNAPP_env 
echo "Conda environment 'SNAPP_env' activated."

SRC_LANG="hsb"
TGT_LANG="de"
MODEL="cis-lmu/glot500-base"
# MODEL="xlm-roberta-base"
# MODEL="sentence-transformers/LaBSE"

case "$MODEL" in
  *glot500*)
    MODEL_TAG="glot500"
    ;;
  *xlm-roberta*|*xlmr*)
    MODEL_TAG="xlmr"
    ;;
  *LaBSE*|*labse*)
    MODEL_TAG="LaBSE"
    ;;
  *)
    MODEL_TAG=$(basename "$MODEL" | tr '/' '_')
    ;;
esac

SCRIPT_PATH="../mean_vectors/create_static_mean_vector.py"
DATA_DIR="../data/${SRC_LANG}-${TGT_LANG}"

SRC_TRAIN="$DATA_DIR/${SRC_LANG}-${TGT_LANG}.train.${SRC_LANG}"
TGT_TRAIN="$DATA_DIR/${SRC_LANG}-${TGT_LANG}.train.${TGT_LANG}"

SRC_OUT="../mean_vectors/mean_vector_${MODEL_TAG}_${SRC_LANG}.txt"
TGT_OUT="../mean_vectors/mean_vector_${MODEL_TAG}_${TGT_LANG}.txt"

echo "start mean vector computation for ${SRC_LANG^^} ($MODEL_TAG)"
python -u "$SCRIPT_PATH" \
  --input_file_path "$SRC_TRAIN" \
  --output_file_path "$SRC_OUT" \
  --model_name "$MODEL"

echo "start mean vector computation for ${TGT_LANG^^} ($MODEL_TAG)"
python -u "$SCRIPT_PATH" \
  --input_file_path "$TGT_TRAIN" \
  --output_file_path "$TGT_OUT" \
  --model_name "$MODEL"

echo "end-time: $(date)"
