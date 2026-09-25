import os
import subprocess

# NOTE: change here
MODEL_NAME = 'glot500'
SRC_LANG = 'chv' 
TRG_LANG = 'ru'
BUCC_SETS = ['train', 'test']

MODEL_MAP = {
    'xlmr': 'xlm-roberta-base',
    'labse': 'sentence-transformers/LaBSE',
    'glot500': 'cis-lmu/glot500-base'
}
MODEL_PATH_HF = MODEL_MAP[MODEL_NAME]

DATA_DIR = '../data'
RESULTS_DIR = f'../results/results_full_{MODEL_NAME}_{SRC_LANG}-{TRG_LANG}'
MINING_DIR = os.path.join(RESULTS_DIR, 'mining', 'bucc2017', f'{SRC_LANG}-{TRG_LANG}')

print(f"Starting SimAlign Post-Processing for model: {MODEL_NAME}")
print("-" * 50)


for data_split in BUCC_SETS:
    
    # input file containing initial candidate pairs (e.g., from PaSeMiLL mining pipeline)
    mapping_file = f"{MINING_DIR}/{MODEL_NAME}.{SRC_LANG}-{TRG_LANG}.{data_split}.sim.pred"
    
    # dataset files (e.g., from BELOPSEM dataset)
    src_file = f"{DATA_DIR}/{SRC_LANG}-{TRG_LANG}/{SRC_LANG}-{TRG_LANG}.{data_split}.{SRC_LANG}"
    trg_file = f"{DATA_DIR}/{SRC_LANG}-{TRG_LANG}/{SRC_LANG}-{TRG_LANG}.{data_split}.{TRG_LANG}"
    gold_file = f"{DATA_DIR}/{SRC_LANG}-{TRG_LANG}/{SRC_LANG}-{TRG_LANG}.{data_split}.gold"
    
    # output files 
    output_file_postprocessing = f"{MINING_DIR}/{MODEL_NAME}.{SRC_LANG}-{TRG_LANG}.{data_split}.sim.pred.postprocessing"
    output_file_postprocessing_res = f"{MINING_DIR}/{MODEL_NAME}.{SRC_LANG}-{TRG_LANG}.{data_split}.sim.pred.postprocessing.res"

    # NOTE: comment arguments to disable them + change hyperparameters if necessary
    command = (
        f"python ../scripts/post-processing_simalign.py "
        f"--mapping-file {mapping_file} "
        f"--src-file {src_file} "
        f"--trg-file {trg_file} "
        f"--model-path {MODEL_PATH_HF} "
        f"--output-file {output_file_postprocessing} "
        f"--window-size 7 "
        f"--min-segment-length 0.10 "
        f"--segment-threshold 0.06 "
        f"--filtering-threshold 0.34 "
        f"--matching-method itermax "   # inter, itermax, mwmf
        f"--token-type bpe "            # bpe, word
        f"--mean-subtraction "          # comment this argument to disable the mean-vector subtraction
        f"--dynamic-mean "              # comment this argument to change from dynamic mean-vector subtraction to the static configuration
        f"--filter-stopwords-src "      # comment this argument to disable the stop-word filtering for the source language
        f"--filter-stopwords-trg "      # comment this argument to disable the stop-word filtering for the target language
        f"--exclude-punctuation "       # comment this argument to disable the punctuation handling
    )
    
    print(f"Running SimAlign Post-Processing on {data_split} set...")
    subprocess.run(command, shell=True, check=True)
    
    eval_command = f"python ../code/scripts/bucc_f-score.py -p {output_file_postprocessing} -g {gold_file} > {output_file_postprocessing_res}"
    subprocess.run(eval_command, shell=True, check=True)
    
    with open(output_file_postprocessing_res, 'r') as f:
        print(f.read())

print("\n\nSimAlign Post-Processing Pipeline complete!")