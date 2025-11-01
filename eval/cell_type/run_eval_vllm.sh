#!/bin/bash
set -eu

# === Environment Setup ===
export CUDA_VISIBLE_DEVICES=0,1

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vLLM

cd ~/cell2sentence/eval/cell_type

# ========================= Configuration =========================

# Base configuration
BASE_DIR="/gpfs/Mamba/Project/Single_Cell/Benchmark/Cell_Type"
DATA_DIR="${BASE_DIR}/Cell2Sentence/Processed_Data"
OUTPUT_DIR="${BASE_DIR}/Cell2Sentence"

# Model configuration
MODEL_PATH="/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Scale-Gemma-2-2B/snapshots/7fc451a816ba12d47c85c5c5ad0036c994705d1f"
# MODEL_PATH="vandijklab/C2S-Scale-Gemma-2-2B"

# Data preparation parameters
N_GENES=200
SEED=1234
ORGANISM="Homo sapiens"

# vLLM configuration
BATCH_SIZE=256
MAX_NEW_TOKENS=20
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=-1

# Datasets to evaluate
DATASETS=(
    "A013:A013_processed_sampled_w_cell2sentence.h5ad"
    # "D099:D099_processed_w_cell2sentence.h5ad"
)

echo "============================================================"
echo "Cell2Sentence vLLM Pipeline"
echo "============================================================"
echo "Data Directory:            ${DATA_DIR}"
echo "Output Directory:          ${OUTPUT_DIR}"
echo "Model Path:                ${MODEL_PATH}"
echo "N Genes:                   ${N_GENES}"
echo "Seed:                      ${SEED}"
echo "Organism:                  ${ORGANISM}"
echo "Batch Size:                ${BATCH_SIZE}"
echo "Max New Tokens:            ${MAX_NEW_TOKENS}"
echo "Tensor Parallel Size:      ${TENSOR_PARALLEL_SIZE}"
echo "GPU Memory Utilization:    ${GPU_MEMORY_UTILIZATION}"
echo "Temperature:               ${TEMPERATURE}"
echo "============================================================"

# Run pipeline for each dataset
for dataset_info in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_id data_file <<< "$dataset_info"
    data_path="${DATA_DIR}/${data_file}"
    dataset_output_dir="${OUTPUT_DIR}/${dataset_id}"
    jsonl_dir="${dataset_output_dir}/jsonl"
    eval_results_dir="${dataset_output_dir}/eval_results/singlecell_openended"
    
    echo ""
    echo "============================================================"
    echo "Processing Dataset: ${dataset_id}"
    echo "============================================================"
    echo "Data Path:             ${data_path}"
    echo "JSONL Dir:             ${jsonl_dir}"
    echo "Eval Results Dir:      ${eval_results_dir}"
    echo ""
    
    if [ ! -f "$data_path" ]; then
        echo "[ERROR] Data file not found: ${data_path}"
        continue
    fi
    
    # Create output directories
    mkdir -p "${jsonl_dir}"
    mkdir -p "${eval_results_dir}"
    
    # Step 1: Convert h5ad to JSONL
    jsonl_file="${jsonl_dir}/${dataset_id}_conversations.jsonl"
    
    if [ ! -f "$jsonl_file" ]; then
        echo "[STEP 1] Converting h5ad to JSONL format..."
        python c2s_h5ad_to_jsonl.py \
            --data_path "${data_path}" \
            --output_file "${jsonl_file}" \
            --dataset_id "${dataset_id}" \
            --n_genes ${N_GENES} \
            --organism "${ORGANISM}" \
            --seed ${SEED}
        
        if [ $? -ne 0 ]; then
            echo "[ERROR] h5ad to JSONL conversion failed for ${dataset_id}"
            continue
        fi
    else
        echo "[STEP 1] JSONL file already exists: ${jsonl_file}"
    fi
    
    # Step 2: Run vLLM inference
    echo "[STEP 2] Running vLLM inference..."
    python c2s_predict_vllm.py \
        --input_file "${jsonl_file}" \
        --output_dir "${eval_results_dir}" \
        --model_path "${MODEL_PATH}" \
        --dataset_id "${dataset_id}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --batch_size ${BATCH_SIZE} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K}
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] vLLM inference failed for ${dataset_id}"
        continue
    fi
    
    # Find the most recent predictions file
    pred_file=$(ls -t "${eval_results_dir}"/singlecell_openended_predictions_vllm_*.json 2>/dev/null | head -1)
    
    if [ -z "$pred_file" ]; then
        echo "[ERROR] No predictions file found in ${eval_results_dir}"
        continue
    fi
    
    echo "[INFO] Found predictions: ${pred_file}"
    
    # Step 3: Apply standardization
    echo "[STEP 3] Applying cell type standardization..."
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

echo ""
echo "============================================================"
echo "Pipeline Completed!"
echo "============================================================"

