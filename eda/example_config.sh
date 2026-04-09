#!/bin/bash

# Example Configuration for Cell2Sentence Embedding Analysis
# Copy this file and modify the paths for your specific dataset and model

# ============================================
# Data Configuration
# ============================================

# Path to your h5ad file
export DATA_PATH="/path/to/your/dataset.h5ad"

# Path to your Cell2Sentence model
export MODEL_PATH="/path/to/your/cell2sentence/model"

# Model name for organizing outputs
export MODEL_NAME="cell2sentence_model_v1"

# ============================================
# Metadata Column Names
# ============================================

# Column name in adata.obs for batch/dataset information
export BATCH_KEY="dataset"

# Column name in adata.obs for cell type information
export CELL_TYPE_KEY="cell_type"

# ============================================
# Analysis Parameters
# ============================================

# Number of top expressed genes to use per cell
export N_GENES=200

# Number of top cell types to show in visualization (-1 for all)
export TOP_CELL_TYPES=20

# Number of neighbors for kNN connectivity calculation
export K_NEIGHBORS=15

# Maximum number of cells to analyze (-1 for all)
# Set this to a smaller number (e.g., 5000-10000) for large datasets
export MAX_SAMPLES=-1

# ============================================
# Output Configuration
# ============================================

# Base output directory
export OUTPUT_BASE_DIR="./results"

# Specific output directories (will be created automatically)
export EMBEDDING_OUTPUT_DIR="${OUTPUT_BASE_DIR}/embedding_analysis/${MODEL_NAME}"
export METRICS_OUTPUT_DIR="${OUTPUT_BASE_DIR}/batch_integration/${MODEL_NAME}"

# ============================================
# Example: Uncomment and modify for real use
# ============================================

# Example 1: Local dataset
# export DATA_PATH="/home/scbjtfy/data/pbmc_dataset.h5ad"
# export MODEL_PATH="/home/scbjtfy/models/c2s_pbmc_finetuned"
# export MODEL_NAME="c2s_pbmc_v1"
# export BATCH_KEY="donor"
# export CELL_TYPE_KEY="cell_type"
# export MAX_SAMPLES=10000

# Example 2: Server dataset
# export DATA_PATH="/data/single_cell/immune_atlas.h5ad"
# export MODEL_PATH="/gpfs/models/cell2sentence/checkpoint-5000"
# export MODEL_NAME="c2s_immune_atlas"
# export BATCH_KEY="dataset"
# export CELL_TYPE_KEY="celltype"
# export MAX_SAMPLES=-1

echo "Configuration loaded successfully!"
echo "Data path: ${DATA_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "Model name: ${MODEL_NAME}"
