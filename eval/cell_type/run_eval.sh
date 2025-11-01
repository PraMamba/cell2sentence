#!/bin/bash
set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl
cd ~/cell2sentence/eval/cell_type

# Configuration
BASE_DIR="/gpfs/Mamba/Project/Single_Cell/Benchmark/Cell_Type"
DATA_DIR="${BASE_DIR}/Cell2Sentence/Processed_Data"
OUTPUT_DIR="${BASE_DIR}/Cell2Sentence"

# Model configuration
# MODEL_PATH="/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Pythia-410m-cell-type-prediction/snapshots/5a4dc3b949b5868ca63752b37bc22e3b0216e435"
# MODEL_TYPE="pythia"

MODEL_PATH="/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Scale-Gemma-2-2B/snapshots/7fc451a816ba12d47c85c5c5ad0036c994705d1f"
MODEL_TYPE="gemma"

# MODEL_PATH="vandijklab/C2S-Scale-Gemma-2-2B"
# MODEL_TYPE="gemma"

# Prediction parameters
N_GENES=200
SEED=1234
ORGANISM="Homo sapiens"

# Datasets to evaluate
DATASETS=(
    "A013:A013_processed_sampled_w_cell2sentence.h5ad"
    # "D099:D099_processed_w_cell2sentence.h5ad"
)

echo "=========================================="
echo "Cell2Sentence Pipeline"
echo "=========================================="
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "Model Type: ${MODEL_TYPE}"
echo "N Genes: ${N_GENES}"
echo "Seed: ${SEED}"
echo "Organism: ${ORGANISM}"
echo "=========================================="

# Run pipeline for each dataset
for dataset_info in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_id data_file <<< "$dataset_info"
    data_path="${DATA_DIR}/${data_file}"
    eval_results_dir="${OUTPUT_DIR}/${dataset_id}/eval_results/singlecell_openended"
    
    echo ""
    echo "=========================================="
    echo "Processing Dataset: ${dataset_id}"
    echo "=========================================="
    echo "Data Path: ${data_path}"
    echo "Output Dir: ${eval_results_dir}"
    echo ""
    
    if [ ! -f "$data_path" ]; then
        echo "[ERROR] Data file not found: ${data_path}"
        continue
    fi
    
    # Create output directory
    mkdir -p "${eval_results_dir}"
    
    # Step 1: Run prediction (directly to eval_results directory)
    echo "[STEP 1] Running Cell2Sentence prediction..."
    python c2s_predict.py \
        --data_path "${data_path}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${eval_results_dir}" \
        --dataset_id "${dataset_id}" \
        --n_genes ${N_GENES} \
        --seed ${SEED} \
        --model_type "${MODEL_TYPE}" \
        --organism "${ORGANISM}"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Prediction failed for ${dataset_id}"
        continue
    fi
    
    # Find the most recent predictions file
    pred_file=$(ls -t "${eval_results_dir}"/singlecell_openended_predictions_*.json 2>/dev/null | head -1)
    
    if [ -z "$pred_file" ]; then
        echo "[ERROR] No predictions file found in ${eval_results_dir}"
        continue
    fi
    
    echo "[INFO] Found predictions: ${pred_file}"
    
    # Step 2: Apply standardization (same directory)
    echo "[STEP 2] Applying cell type standardization..."
    python singlecell_openended_eval.py \
        --predictions_file "${pred_file}" \
        --output_dir "${eval_results_dir}"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed ${dataset_id}"
        echo "[INFO] Results saved to: ${eval_results_dir}"
    else
        echo "[ERROR] Standardization failed for ${dataset_id}"
    fi
done

echo "=========================================="
echo "Pipeline Completed!"

