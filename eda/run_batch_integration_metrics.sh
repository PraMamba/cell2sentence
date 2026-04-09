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
OUTPUT_DIR="./results/batch_integration/${MODEL_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Analysis parameters
BATCH_KEY="dataset"        # Column name in h5ad.obs for batch/dataset information
CELL_TYPE_KEY="cell_type"  # Column name in h5ad.obs for cell type information
N_GENES=200                # Number of top expressed genes per cell
K_NEIGHBORS=15             # Number of neighbors for kNN connectivity calculation
MAX_SAMPLES=-1             # Set to -1 to use all samples, or specify max number to sample

# === Log Setup ===
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/batch_integration_${DATE_SUFFIX}.log"

echo "=============================================="
echo "Cell2Sentence Batch Integration Metrics (scIB)"
echo "=============================================="
echo "Data Path: ${DATA_PATH}"
echo "Model Path: ${MODEL_PATH}"
echo "Batch Key: ${BATCH_KEY}"
echo "Cell Type Key: ${CELL_TYPE_KEY}"
echo "N Genes: ${N_GENES}"
echo "K Neighbors: ${K_NEIGHBORS}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

# === Run Analysis ===
echo "Starting analysis at $(date)"
echo "Starting analysis at $(date)" >> "${LOG_FILE}" 2>&1

python3 calculate_c2s_batch_integration_metrics.py \
    --data_path "${DATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_key "${BATCH_KEY}" \
    --cell_type_key "${CELL_TYPE_KEY}" \
    --n_genes ${N_GENES} \
    --k_neighbors ${K_NEIGHBORS} \
    --max_samples ${MAX_SAMPLES} \
    >> "${LOG_FILE}" 2>&1

echo "Analysis completed at $(date)"
echo "Analysis completed at $(date)" >> "${LOG_FILE}" 2>&1

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
