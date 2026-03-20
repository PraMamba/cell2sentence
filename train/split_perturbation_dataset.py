#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split Perturbation Dataset into Train and Test Sets

This script splits a JSONL file containing perturbation data into training and test sets
based on a list of test perturbation IDs.

Usage:
    python split_perturbation_dataset.py \
        --input_jsonl /path/to/input.jsonl \
        --test_ids_file /path/to/test_ids.txt \
        --output_dir /path/to/output \
        [--train_output train.jsonl] \
        [--test_output test.jsonl]
"""

import os
import json
import argparse
from typing import Set
from tqdm import tqdm


def load_test_ids(test_ids_file: str) -> Set[str]:
    """
    Load test perturbation IDs from a text file.
    
    Args:
        test_ids_file: Path to file containing test IDs (one per line)
    
    Returns:
        Set of test perturbation IDs
    """
    print(f"Loading test IDs from: {test_ids_file}")
    test_ids = set()
    
    with open(test_ids_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                test_ids.add(line)
    
    print(f"Loaded {len(test_ids)} test IDs")
    return test_ids


def split_dataset(
    input_jsonl: str,
    test_ids: Set[str],
    train_output: str,
    test_output: str
):
    """
    Split JSONL dataset into train and test sets based on perturbation IDs.
    
    Args:
        input_jsonl: Path to input JSONL file
        test_ids: Set of test perturbation IDs
        train_output: Path to output training JSONL file
        test_output: Path to output test JSONL file
    """
    print(f"\nProcessing input file: {input_jsonl}")
    
    train_count = 0
    test_count = 0
    error_count = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, \
         open(train_output, 'w', encoding='utf-8') as f_train, \
         open(test_output, 'w', encoding='utf-8') as f_test:
        
        # Count total lines for progress bar
        print("Counting total records...")
        total_lines = sum(1 for _ in open(input_jsonl, 'r', encoding='utf-8'))
        
        # Reset file pointer and process
        for line in tqdm(f_in, total=total_lines, desc="Splitting dataset"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Extract perturbation_id from metadata
                perturbation_id = record.get('metadata', {}).get('perturbation_id', '')
                
                if not perturbation_id:
                    print(f"Warning: Record without perturbation_id found: {record.get('id', 'unknown')}")
                    error_count += 1
                    continue
                
                # Decide which set this record belongs to
                if perturbation_id in test_ids:
                    f_test.write(json.dumps(record, ensure_ascii=False) + '\n')
                    test_count += 1
                else:
                    f_train.write(json.dumps(record, ensure_ascii=False) + '\n')
                    train_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                error_count += 1
                continue
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Split Statistics:")
    print("=" * 60)
    print(f"Total records processed: {train_count + test_count}")
    print(f"Training set: {train_count} records")
    print(f"Test set: {test_count} records")
    print(f"Errors/Skipped: {error_count} records")
    print(f"\nTrain output: {train_output}")
    print(f"Test output: {test_output}")
    print("=" * 60)
    
    # Check for missing test IDs
    print("\nVerifying test ID coverage...")
    with open(test_output, 'r', encoding='utf-8') as f:
        found_test_ids = set()
        for line in f:
            record = json.loads(line.strip())
            perturbation_id = record.get('metadata', {}).get('perturbation_id', '')
            found_test_ids.add(perturbation_id)
    
    missing_ids = test_ids - found_test_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} test IDs not found in input file:")
        for missing_id in sorted(list(missing_ids))[:10]:  # Show first 10
            print(f"  - {missing_id}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more")
    else:
        print("All test IDs were found in the input file ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Split perturbation dataset into train and test sets"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--test_ids_file",
        type=str,
        required=True,
        help="Path to file containing test perturbation IDs (one per line)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--train_output",
        type=str,
        default="train.jsonl",
        help="Name of training output file (default: train.jsonl)"
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="test.jsonl",
        help="Name of test output file (default: test.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")
    
    if not os.path.exists(args.test_ids_file):
        raise FileNotFoundError(f"Test IDs file not found: {args.test_ids_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct full output paths
    train_output_path = os.path.join(args.output_dir, args.train_output)
    test_output_path = os.path.join(args.output_dir, args.test_output)
    
    # Load test IDs
    test_ids = load_test_ids(args.test_ids_file)
    
    # Split dataset
    split_dataset(
        input_jsonl=args.input_jsonl,
        test_ids=test_ids,
        train_output=train_output_path,
        test_output=test_output_path
    )
    
    print("\nDataset split completed successfully!")


if __name__ == "__main__":
    main()

