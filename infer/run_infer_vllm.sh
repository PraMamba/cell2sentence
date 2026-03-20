#!/bin/bash

set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy

conda activate vLLM
cd ~/cell2sentence/infer

export CUDA_VISIBLE_DEVICES=6,7
export VLLM_USE_V1=1

MODEL_PATH="/data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_Perturbation_Response-P021_Tahoe1000/2025-12-26-23_30_00_finetune_perturbation_prediction/checkpoint-10500"
INPUT_FILE="${INPUT_FILE:-/data/Mamba/Project/Single_Cell/Data/RVQ-NC=8-CS=32/Perturbation/STATIC=1-MAX_GENES=-1-full_random-P021_Tahoe1000-C2S/test_conversation.jsonl}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_Perturbation_Response-P021_Tahoe1000/2025-12-26-23_30_00_finetune_perturbation_prediction/checkpoint-10500/infer}"

BATCH_SIZE=8192
MAX_NEW_TOKENS=4096
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=-1

OUTPUT_DIR="${BASE_OUTPUT_DIR}/inference-vllm_$(date +%Y%m%d_%H)_MAX_NEW_TOKENS-${MAX_NEW_TOKENS}_TEMPERATURE-${TEMPERATURE}_TOP_P-${TOP_P}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Cell2Sentence vLLM Inference"
echo "============================================================"
echo "Start time: $(date)"
echo "Configuration:"
echo "----------------------------------------------"
echo "Model Path:               ${MODEL_PATH}"
echo "Input File:               ${INPUT_FILE}"
echo "Output Directory:         ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES:     ${CUDA_VISIBLE_DEVICES}"
echo "Tensor Parallel Size:     ${TENSOR_PARALLEL_SIZE}"
echo "Batch Size:               ${BATCH_SIZE}"
echo "Max New Tokens:           ${MAX_NEW_TOKENS}"
echo "GPU Memory Utilization:   ${GPU_MEMORY_UTILIZATION}"
echo "Temperature:              ${TEMPERATURE}"
echo "Top-p:                    ${TOP_P}"
echo "Top-k:                    ${TOP_K}"
echo "============================================================"

python3 c2s_predict_vllm.py \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K}

