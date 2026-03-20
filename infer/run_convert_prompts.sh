#!/bin/bash

set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy

conda activate vLLM
cd ~/cell2sentence/infer

# === Configuration ===
INPUT_FILE="${INPUT_FILE:-/data/Mamba/Project/Single_Cell/Data/Perturbation/STATIC=1-MAX_GENES=-1-full_random/perturbation_topgenes_c2s_P020_conversation_train.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${INPUT_FILE%.jsonl}_c2s.jsonl}"
NUM_GENES="${NUM_GENES:-500}"  # Number of genes to use (default: 500)

echo "============================================================"
echo "Convert Test Prompts to Training Format"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "----------------------------------------------"
echo "Input File:               ${INPUT_FILE}"
echo "Output File:              ${OUTPUT_FILE}"
echo "Number of Genes:          ${NUM_GENES}"
echo "============================================================"
echo ""

# Run conversion
python3 convert_test_prompts.py \
    --input_file "${INPUT_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --num_genes ${NUM_GENES}

echo ""
echo "============================================================"
echo "End time: $(date)"
echo "============================================================"

