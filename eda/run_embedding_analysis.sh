#!/bin/bash

set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

cd ~/cell2sentence/eda

# === Configuration ===
DATE_SUFFIX=$(date +"%Y%m%d_%H")

# Input paths - MODIFY THESE FOR YOUR DATA
DATA_PATH="/path/to/your/dataset.h5ad"
MODEL_NAME="cell2sentence_model"
MODEL_PATH="/path/to/your/cell2sentence/model"

# Output configuration
OUTPUT_DIR="./results/embedding_analysis/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Analysis parameters
N_GENES=200              # Number of top expressed genes per cell
TOP_CELL_TYPES=-1        # Set to -1 to show all cell types, or specify a number for top N
MAX_SAMPLES=-1           # Set to -1 to use all samples, or specify max number to sample

# === Log Setup ===
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/embedding_analysis_${DATE_SUFFIX}.log"

echo "=============================================="
echo "Cell2Sentence Embedding Distribution Analysis"
echo "=============================================="
echo "Data Path: ${DATA_PATH}"
echo "Model Path: ${MODEL_PATH}"
echo "N Genes: ${N_GENES}"
echo "Top Cell Types: ${TOP_CELL_TYPES}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

# === Run Analysis ===
echo "Starting analysis at $(date)"
echo "Starting analysis at $(date)" >> "${LOG_FILE}" 2>&1

python3 analyze_c2s_embedding_distribution.py \
    --data_path "${DATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_genes ${N_GENES} \
    --top_cell_types ${TOP_CELL_TYPES} \
    --max_samples ${MAX_SAMPLES} \
    >> "${LOG_FILE}" 2>&1

echo "Analysis completed at $(date)"
echo "Analysis completed at $(date)" >> "${LOG_FILE}" 2>&1

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
