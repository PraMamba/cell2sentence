#!/bin/bash
# -*- coding: utf-8 -*-
#=============================================================================
# Split Perturbation Dataset Script
#=============================================================================
# This script splits the perturbation response dataset into training and test sets
# based on a list of test perturbation IDs.
#
# Usage:
#   bash run_split_dataset.sh
#=============================================================================

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
unset http_proxy https_proxy all_proxy

conda activate vLLM

# === Navigate to Project Directory ===
cd ~/cell2sentence

#=============================================================================
# Configuration
#=============================================================================

# Input files
INPUT_JSONL="/data/Mamba/Project/Single_Cell/Data/Perturbation/STATIC=1-MAX_GENES=-1-full_random/perturbation_topgenes_c2s_P020_conversation.jsonl"
TEST_IDS_FILE="/home/scbjtfy/cell2sentence/test_30ids.txt"

# Output directory
OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Data/Perturbation/STATIC=1-MAX_GENES=-1-full_random/split"

# Output file names
TRAIN_OUTPUT="train_conversation.jsonl"
TEST_OUTPUT="test_conversation.jsonl"

# Python environment (optional, uncomment if needed)
# CONDA_ENV="vLLM"
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate $CONDA_ENV

#=============================================================================
# Validation
#=============================================================================

echo "=============================================="
echo "Cell2Sentence Dataset Split"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Check if input files exist
if [ ! -f "$INPUT_JSONL" ]; then
    echo "Error: Input JSONL file not found: $INPUT_JSONL"
    exit 1
fi

if [ ! -f "$TEST_IDS_FILE" ]; then
    echo "Error: Test IDs file not found: $TEST_IDS_FILE"
    echo ""
    echo "Please create the test IDs file with one perturbation ID per line."
    echo "Example format:"
    echo "  P013_Open_Problems_NK cells_donor_2_Protriptyline_1.0uM"
    echo "  P013_Open_Problems_NK cells_donor_0_Phenylbutazone_1.0uM"
    echo "  ..."
    exit 1
fi

#=============================================================================
# Display Configuration
#=============================================================================

echo "Configuration:"
echo "----------------------------------------------"
echo "Input JSONL:     $INPUT_JSONL"
echo "Test IDs file:   $TEST_IDS_FILE"
echo "Output dir:      $OUTPUT_DIR"
echo "Train output:    $TRAIN_OUTPUT"
echo "Test output:     $TEST_OUTPUT"
echo "----------------------------------------------"
echo ""

# Display file statistics
echo "Input file statistics:"
TOTAL_RECORDS=$(wc -l < "$INPUT_JSONL")
TEST_IDS_COUNT=$(wc -l < "$TEST_IDS_FILE")
echo "  Total records in input: $TOTAL_RECORDS"
echo "  Test IDs to extract:    $TEST_IDS_COUNT"
echo ""

# Show first few test IDs
echo "First 5 test IDs:"
head -n 5 "$TEST_IDS_FILE" | sed 's/^/  - /'
echo ""

# Confirm before proceeding (optional, comment out for automation)
# read -p "Proceed with dataset split? (y/n) " -n 1 -r
# echo
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Operation cancelled."
#     exit 0
# fi

#=============================================================================
# Run Dataset Split
#=============================================================================

echo "Starting dataset split..."
echo "=============================================="
echo ""

python3 ./train/split_perturbation_dataset.py \
    --input_jsonl "$INPUT_JSONL" \
    --test_ids_file "$TEST_IDS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --train_output "$TRAIN_OUTPUT" \
    --test_output "$TEST_OUTPUT"

#=============================================================================
# Post-processing and Verification
#=============================================================================

echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

# Check if output files were created
if [ -f "$OUTPUT_DIR/$TRAIN_OUTPUT" ] && [ -f "$OUTPUT_DIR/$TEST_OUTPUT" ]; then
    TRAIN_RECORDS=$(wc -l < "$OUTPUT_DIR/$TRAIN_OUTPUT")
    TEST_RECORDS=$(wc -l < "$OUTPUT_DIR/$TEST_OUTPUT")
    
    echo "✓ Split completed successfully!"
    echo ""
    echo "Output statistics:"
    echo "  Training set:   $TRAIN_RECORDS records"
    echo "  Test set:       $TEST_RECORDS records"
    echo "  Total:          $((TRAIN_RECORDS + TEST_RECORDS)) records"
    echo ""
    echo "Output files:"
    echo "  Train: $OUTPUT_DIR/$TRAIN_OUTPUT"
    echo "  Test:  $OUTPUT_DIR/$TEST_OUTPUT"
    echo ""
    
    # Calculate split ratio
    TRAIN_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($TRAIN_RECORDS / ($TRAIN_RECORDS + $TEST_RECORDS)) * 100}")
    TEST_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($TEST_RECORDS / ($TRAIN_RECORDS + $TEST_RECORDS)) * 100}")
    echo "Split ratio: ${TRAIN_PERCENT}% train / ${TEST_PERCENT}% test"
    
    # Show sample from each file
    echo ""
    echo "Sample from training set (first record ID):"
    head -n 1 "$OUTPUT_DIR/$TRAIN_OUTPUT" | jq -r '.id' | sed 's/^/  /'
    
    echo ""
    echo "Sample from test set (first record ID):"
    head -n 1 "$OUTPUT_DIR/$TEST_OUTPUT" | jq -r '.id' | sed 's/^/  /'
    
else
    echo "✗ Error: Output files were not created successfully"
    exit 1
fi

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

