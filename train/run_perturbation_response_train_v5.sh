#!/bin/bash
#
# Train C2S (Cell2Sentence) Gemma-2-2B on v5 benchmark topgenes data.
#
# Input:  /data/scbank/size32/v5_benchmark/topgenes/conversation_dataset_train/conversation.jsonl
# Output: /data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_v5_Perturb_topgenes_{DATE}/
#
# Requires: conda activate vLLM (for liger_kernel etc.)
# GPUs: 2 GPUs needed (Gemma-2-2B with batch=1, max_length=8192)

set -eu

export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT="RVQ_Alpha"
export SWANLAB_WORKSPACE="GZLab-Tian"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy 2>/dev/null || true
conda activate vLLM

cd ~/cell2sentence

# === Experiment Parameters ===
LAUNCH_TYPE=DeepSpeed
MODEL_TYPE=C2S-Scale-Gemma-2-2B-v5-Tahoe500
MODEL_PATH=/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Scale-Gemma-2-2B/snapshots/7fc451a816ba12d47c85c5c5ad0036c994705d1f

INPUT_FILE="/data/scbank/size32/v5_benchmark/topgenes/conversation_dataset_train/conversation.jsonl"

NUM_GENES=500
SEED=42
VAL_SPLIT=0.1
DATE_SUFFIX=$(date +"%Y%m%d_%H")
OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_v5_Perturb_topgenes_${DATE_SUFFIX}"
RUN_NAME="${MODEL_TYPE}_Perturbation_${DATE_SUFFIX}"

# GPU assignment — update these based on availability
GPU_DEVICES="${GPU_DEVICES:-localhost:6,7}"
MASTER_PORT="${MASTER_PORT:-29503}"

# Training hyperparameters
NUM_EPOCHS=3
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=2
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
MAX_LENGTH=8192
SAVE_STEPS=500
EVAL_STEPS=500
LOGGING_STEPS=1
SAVE_LIMIT=3
NUM_WORKERS=0

# === Log Setup ===
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training_${DATE_SUFFIX}.log"

echo "=============================================="
echo "C2S v5 Perturbation Response Training"
echo "=============================================="
echo "Start time: $(date)"
echo "Model: ${MODEL_PATH}"
echo "Data: ${INPUT_FILE}"
echo "GPUs: ${GPU_DEVICES}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=============================================="
echo ""
echo "Monitor: tail -f ${LOG_FILE}"
echo "=============================================="

DEEPSPEED_CONFIG=./train/configs/ds_zero2.json

common_args="
    --input_file $INPUT_FILE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --num_genes $NUM_GENES \
    --seed $SEED \
    --val_split $VAL_SPLIT \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --max_length $MAX_LENGTH \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_LIMIT \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_strategy steps \
    --logging_steps $LOGGING_STEPS \
    --logging_first_step True \
    --dataloader_num_workers $NUM_WORKERS \
    --bf16 True \
    --fp16 False \
    --gradient_checkpointing False \
    --ddp_find_unused_parameters False \
    --ddp_timeout 30000 \
    --log_on_each_node False \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --report_to swanlab \
    --use_liger_kernel True \
"

deepspeed \
    --include ${GPU_DEVICES} \
    --master_port ${MASTER_PORT} \
    ./train/perturbation_response_train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    ${common_args} \
    >> "${LOG_FILE}" 2>&1
