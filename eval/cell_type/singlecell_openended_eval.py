"""
Single-cell Open-ended Standardization Script for Cell2Sentence

This script standardizes Cell2Sentence predictions using cell type mappings
and tracks unmapped cell types.

Usage:
    python singlecell_openended_eval.py \
        --predictions_file /path/to/predictions.json \
        --output_dir /path/to/save/results
"""

import os
import json
import argparse
from typing import List, Dict
from datetime import datetime

from celltype_standardizer import CellTypeStandardizer, save_unmapped_report


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} predictions from {predictions_file}")
    return data


def standardize_predictions(
    predictions: List[Dict],
    standardizer: CellTypeStandardizer
) -> tuple:
    """
    Standardize cell type names in predictions using mapping.
    
    Args:
        predictions: List of prediction dictionaries
        standardizer: CellTypeStandardizer object
        
    Returns:
        Tuple of (standardized_predictions, unmapped_records)
    """
    print("[INFO] Standardizing cell type names...")
    
    standardized_predictions = []
    unmapped_records = []
    
    for pred in predictions:
        idx = pred["index"]
        
        # Standardize ground truth
        gt_raw = pred["ground_truth"]
        gt_std, gt_is_mapped = standardizer.standardize_single_celltype(gt_raw)
        
        # Standardize predicted answer
        pred_raw = pred["predicted_answer"]
        pred_std, pred_is_mapped = standardizer.standardize_single_celltype(pred_raw)
        
        # Track unmapped types
        if not gt_is_mapped and gt_raw:
            unmapped_records.append({
                "index": idx,
                "source": "ground_truth",
                "original_type": gt_raw,
                "full_answer": gt_raw
            })
        
        if not pred_is_mapped and pred_raw:
            unmapped_records.append({
                "index": idx,
                "source": "predicted_answer",
                "original_type": pred_raw,
                "full_answer": pred_raw
            })
        
        # Create standardized prediction (without _raw fields)
        std_pred = pred.copy()
        std_pred["ground_truth"] = gt_std
        std_pred["predicted_answer"] = pred_std
        
        standardized_predictions.append(std_pred)
    
    print(f"[INFO] Standardization complete")
    print(f"[INFO] Found {len(unmapped_records)} unmapped instances")
    
    return standardized_predictions, unmapped_records


def save_results(
    standardized_predictions: List[Dict],
    predictions_file: str
):
    """
    Save standardized predictions by overwriting the input file.
    
    Args:
        standardized_predictions: List of standardized prediction dictionaries
        predictions_file: Path to the predictions file (will be overwritten)
    """
    # Overwrite the original predictions file with standardized version
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(standardized_predictions, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Standardized predictions saved to: {predictions_file}")


def run_evaluation(predictions_file: str, output_dir: str):
    """
    Run standardization pipeline.
    
    Args:
        predictions_file: Path to predictions JSON file (will be overwritten with standardized version)
        output_dir: Directory to save unmapped cell types report
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load predictions
    predictions = load_predictions(predictions_file)
    
    # Initialize standardizer
    standardizer = CellTypeStandardizer()
    
    # Standardize predictions
    standardized_predictions, unmapped_records = standardize_predictions(
        predictions,
        standardizer
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("STANDARDIZATION RESULTS")
    print("=" * 80)
    print(f"Total Cells:      {len(standardized_predictions)}")
    print(f"Unmapped Types:   {len(unmapped_records)}")
    print("=" * 80)
    
    # Save standardized predictions (overwrite input file)
    save_results(standardized_predictions, predictions_file)
    
    # Save unmapped report
    save_unmapped_report(
        unmapped_records,
        output_dir,
        "singlecell_openended",
        timestamp
    )
    
    print("\n[SUCCESS] Standardization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Standardize Cell2Sentence predictions using cell type mappings"
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_file}")
    
    run_evaluation(
        predictions_file=args.predictions_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

