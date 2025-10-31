"""
Convert Notebook Predictions to Standardized Format

This script converts predictions from the Cell2Sentence notebook (CSV format)
to the standardized JSON format used by the evaluation pipeline.

Usage:
    python convert_notebook_predictions.py \
        --csv_file /path/to/c2s_A013_predict.csv \
        --h5ad_file /path/to/A013_processed_sampled_w_cell2sentence.h5ad \
        --dataset_id A013 \
        --output_dir ./output
"""

import os
import json
import argparse
from typing import List, Dict
from datetime import datetime

import pandas as pd
import anndata


def load_predictions_csv(csv_file: str) -> pd.DataFrame:
    """Load predictions from CSV file."""
    print(f"[INFO] Loading predictions from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"[INFO] Loaded {len(df)} predictions")
    return df


def load_adata(h5ad_file: str) -> anndata.AnnData:
    """Load AnnData object."""
    print(f"[INFO] Loading AnnData from: {h5ad_file}")
    adata = anndata.read_h5ad(h5ad_file)
    print(f"[INFO] Loaded {adata.shape[0]} cells with {adata.shape[1]} genes")
    return adata


def get_cell_sentence(adata, cell_idx: int, n_genes: int = 200) -> str:
    """
    Get the cell sentence (gene list) that was input to the model.
    
    Args:
        adata: AnnData object
        cell_idx: Index of the cell
        n_genes: Number of top genes used by model
        
    Returns:
        Space-separated gene list (the actual model input)
    """
    import numpy as np
    
    # Get expression values for this cell
    cell_exp = adata.X[cell_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[cell_idx]
    
    # Get indices of top genes
    top_indices = np.argsort(cell_exp)[-n_genes:][::-1]
    
    # Get gene names
    top_genes = adata.var_names[top_indices].tolist()
    
    # Return as space-separated string (this is what C2S model sees)
    return ' '.join(top_genes)




def convert_predictions_to_json(
    df: pd.DataFrame,
    adata: anndata.AnnData,
    dataset_id: str,
    n_genes: int = 200
) -> List[Dict]:
    """
    Convert CSV predictions to JSON format.
    
    Args:
        df: DataFrame with 'Predicted' and 'GroundTruth' columns
        adata: AnnData object
        dataset_id: Dataset identifier
        n_genes: Number of genes used by model (default: 200)
        
    Returns:
        List of prediction dictionaries in standardized format
    """
    print("[INFO] Converting predictions to JSON format...")
    
    results = []
    
    for idx, row in df.iterrows():
        # Get ground truth and prediction
        ground_truth = row['GroundTruth']
        predicted_answer = row['Predicted']
        
        # Clean prediction (remove trailing period if present)
        if predicted_answer and str(predicted_answer).endswith('.'):
            predicted_answer = str(predicted_answer)[:-1].strip()
        
        # Get the actual model input (cell sentence with top N genes)
        question = get_cell_sentence(adata, idx, n_genes=n_genes)
        
        # The full_response is just the predicted cell type (C2S outputs directly)
        full_response = str(predicted_answer)
        
        # Create result in standardized format
        result = {
            "model_name": "vandijklab/C2S-Pythia-410m-cell-type-prediction",
            "dataset_id": dataset_id,
            "index": int(idx),
            "task_type": "cell type",
            "task_variant": "singlecell_openended",
            "question": question,
            "ground_truth": str(ground_truth),
            "predicted_answer": str(predicted_answer),
            "full_response": full_response,
            "group": ""
        }
        
        results.append(result)
    
    print(f"[INFO] Converted {len(results)} predictions")
    
    return results


def save_predictions(results: List[Dict], output_dir: str) -> str:
    """
    Save predictions to JSON file.
    
    Args:
        results: List of prediction dictionaries
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = os.path.join(
        output_dir,
        f"singlecell_openended_predictions_{timestamp}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Predictions saved to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert notebook predictions to standardized JSON format"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with predictions (e.g., c2s_A013_predict.csv)"
    )
    parser.add_argument(
        "--h5ad_file",
        type=str,
        required=True,
        help="Path to .h5ad file"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="Dataset identifier (e.g., A013, D099)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save converted predictions"
    )
    parser.add_argument(
        "--n_genes",
        type=int,
        default=200,
        help="Number of top genes used by model (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    
    if not os.path.exists(args.h5ad_file):
        raise FileNotFoundError(f"H5AD file not found: {args.h5ad_file}")
    
    print("\n" + "=" * 80)
    print("Converting Notebook Predictions to Standardized Format")
    print("=" * 80)
    print(f"CSV File:    {args.csv_file}")
    print(f"H5AD File:   {args.h5ad_file}")
    print(f"Dataset ID:  {args.dataset_id}")
    print(f"Output Dir:  {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Load data
    df = load_predictions_csv(args.csv_file)
    adata = load_adata(args.h5ad_file)
    
    # Verify lengths match
    if len(df) != adata.shape[0]:
        print(f"[WARNING] CSV has {len(df)} rows but h5ad has {adata.shape[0]} cells")
        print(f"[WARNING] Using minimum length: {min(len(df), adata.shape[0])}")
        df = df.iloc[:min(len(df), adata.shape[0])]
    
    # Convert to JSON
    results = convert_predictions_to_json(df, adata, args.dataset_id, args.n_genes)
    
    # Save
    output_file = save_predictions(results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total predictions: {len(results)}")
    print("=" * 80 + "\n")
    
    print("Next step: Run evaluation")
    print(f"  python singlecell_openended_eval.py \\")
    print(f"      --predictions_file {output_file} \\")
    print(f"      --output_dir ./eval_results")


if __name__ == "__main__":
    main()

