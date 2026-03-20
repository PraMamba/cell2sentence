#!/bin/bash

set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy

conda activate vLLM
cd ~/cell2sentence

INPUT_DIR="/data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_Perturbation_Response/2025-11-20-22_48_46_finetune_perturbation_prediction/checkpoint-8000/infer"
INPUT_FILE="${INPUT_DIR}/predictions_20251121_153924.jsonl"
OUTPUT_DIR="${INPUT_DIR}/eval_results"
GENE_LIST="/home/scbjtfy/RVQ-Alpha/data_utils/gene_name_list_with_index.csv"

mkdir -p "${OUTPUT_DIR}"
CLEANED_FILE="${INPUT_FILE}_predictions_cleaned.jsonl"

echo "============================================================"
echo "Evaluate Top-K Genes"
echo "============================================================"
echo "Start time: $(date)"
echo "Configuration:"
echo "----------------------------------------------"
echo "Input File:               ${INPUT_FILE}"

echo "Running evaluation..."
python ./eval/topk_genes/evaluate_topk_genes.py \
    --mode all \
    --input_file ${INPUT_FILE} \
    --output_file ${CLEANED_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --gene_list ${GENE_LIST}

