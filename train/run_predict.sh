#!/bin/bash
# Inference script for Cell2Sentence perturbation response prediction
# 
# This script runs predictions using a fine-tuned model on test data.

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Model path - MODIFY THIS to point to your trained checkpoint
# Example: ./output/perturbation_training_20250101_120000/2025-01-01-12_00_00_finetune_perturbation_prediction/checkpoint-1000
MODEL_PATH="./output/2025-01-01-12_00_00_finetune_perturbation_prediction/checkpoint-1000"

# Input test data - MODIFY THIS to point to your test data
INPUT_FILE="/data/Mamba/Project/Single_Cell/Data/Perturbation/STATIC=1-MAX_GENES=-1-full_random/perturbation_topgenes_c2s_P020_conversation.jsonl"

# Output file
OUTPUT_FILE="./predictions_$(date +%Y%m%d_%H%M%S).jsonl"

# Prediction parameters
NUM_GENES=500           # Should match training configuration
MAX_TOKENS=2000         # Maximum tokens to generate
BATCH_SIZE=1            # Inference batch size
TOP_K_OVERLAP=50        # For computing overlap metrics

echo "========================================="
echo "Cell2Sentence Perturbation Prediction"
echo "========================================="
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Number of genes: $NUM_GENES"
echo "========================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please specify a valid checkpoint path."
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Run prediction
python predict_perturbation.py \
    --model_path "$MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_genes $NUM_GENES \
    --max_tokens $MAX_TOKENS \
    --batch_size $BATCH_SIZE \
    --compute_metrics \
    --top_k_overlap $TOP_K_OVERLAP

# Check prediction status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Prediction completed successfully!"
    echo "End time: $(date)"
    echo "Results saved to: $OUTPUT_FILE"
    echo "========================================="
else
    echo "========================================="
    echo "Prediction failed with error code $?"
    echo "========================================="
    exit 1
fi

