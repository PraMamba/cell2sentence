#!/bin/bash

set -eu

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false
export SWANLAB_PROJECT="RVQ_Alpha"
export SWANLAB_WORKSPACE="GZLab-Tian"
export SWANLAB_RESUME=True
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-4K6zQipDnSGeRUJACfQD0}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy

conda activate vLLM

# === Navigate to Project Directory ===
cd ~/cell2sentence

# === Experiment Parameters ===
LAUNCH_TYPE=DeepSpeed  # Options: TorchRun, DeepSpeed
MODEL_TYPE=C2S-Scale-Gemma-2-2B-P021_Tahoe1000
MODEL_PATH=/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Scale-Gemma-2-2B/snapshots/7fc451a816ba12d47c85c5c5ad0036c994705d1f

# Data paths
INPUT_FILE="/data/Mamba/Project/Single_Cell/Data/RVQ-NC=8-CS=32/Perturbation/STATIC=1-MAX_GENES=-1-full_random-P021_Tahoe1000-C2S/train_conversation.jsonl"

# Training parameters
NUM_GENES=500
SEED=42
VAL_SPLIT=0.1
DATE_SUFFIX=$(date +"%Y%m%d_%H")
OUTPUT_DIR=/data/Mamba/Project/Single_Cell/Training/Cell2Sentence_Gemma2_2B_Perturbation_Response-P021_Tahoe1000
RUN_NAME=${MODEL_TYPE}_Perturbation

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
MASTER_PORT=29500

# === Log Setup ===
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training_${DATE_SUFFIX}.log"

echo "=============================================="
echo "Cell2Sentence Perturbation Response Training"
echo "=============================================="
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Launch type: $LAUNCH_TYPE"
echo "Logging to: ${LOG_FILE}"
echo "=============================================="
echo ""
echo "Real-Time Training Log Monitoring:"
echo "tail -f ${LOG_FILE}"
echo "=============================================="

# === Common Arguments ===
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

# === Launch Training ===
if [[ "$LAUNCH_TYPE" == "TorchRun" ]]; then
    echo "Using TorchRun"
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    echo "Detected $NUM_GPUS GPUs"
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        ./train/perturbation_response_train.py \
        $common_args \
        >> "$LOG_FILE" 2>&1
        
elif [[ "$LAUNCH_TYPE" == "DeepSpeed" ]]; then
    echo "Using DeepSpeed"
    GPU_DEVICES="localhost:0,1,2,3,4,5,6,7"
    DEEPSPEED_CONFIG=./train/configs/ds_zero2.json
    
    deepspeed \
        --include $GPU_DEVICES \
        --master_port $MASTER_PORT \
        ./train/perturbation_response_train.py \
        --deepspeed $DEEPSPEED_CONFIG \
        $common_args \
        >> "$LOG_FILE" 2>&1
else
    echo "Error: Invalid LAUNCH_TYPE: $LAUNCH_TYPE"
    echo "Valid options: TorchRun, DeepSpeed"
    exit 1
fi
