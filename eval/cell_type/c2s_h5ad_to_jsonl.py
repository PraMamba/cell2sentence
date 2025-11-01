"""
Convert h5ad dataset to conversations format JSONL for vLLM inference

This script converts Cell2Sentence h5ad files to the conversation format
required by vLLM inference pipeline.

Usage:
    python c2s_h5ad_to_jsonl.py \
        --data_path /path/to/dataset.h5ad \
        --output_file /path/to/output.jsonl \
        --dataset_id A013 \
        --n_genes 200 \
        --organism "Homo sapiens" \
        --seed 1234
"""

import os
import json
import argparse
import random
from typing import List, Dict

import numpy as np
import anndata
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs


def set_seed(seed: int = 1234):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_data(data_path: str, seed: int = 1234) -> tuple:
    """
    Load h5ad file and convert to Cell2Sentence format.
    
    Args:
        data_path: Path to .h5ad file
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (arrow_ds, vocabulary, adata)
    """
    print(f"[INFO] Loading data from: {data_path}")
    adata = anndata.read_h5ad(data_path)
    
    # Get observation columns to keep
    adata_obs_cols_to_keep = adata.obs.columns.tolist()
    
    print(f"[INFO] Dataset shape: {adata.shape}")
    print(f"[INFO] Observation columns: {adata_obs_cols_to_keep}")
    
    # Convert to Cell2Sentence format
    print("[INFO] Converting to Cell2Sentence format...")
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata, 
        random_state=seed, 
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )
    
    print(f"[INFO] Created arrow dataset with {len(arrow_ds)} cells")
    
    return arrow_ds, vocabulary, adata


def create_conversation_item(
    cell_sentence: str,
    ground_truth: str,
    n_genes: int,
    organism: str,
    index: int,
    dataset_id: str,
    group: str = ""
) -> Dict:
    """
    Create a single conversation item in the required format.
    
    Args:
        cell_sentence: Space-separated gene names
        ground_truth: Ground truth cell type
        n_genes: Number of genes used
        organism: Organism name
        index: Cell index
        dataset_id: Dataset identifier
        group: Group/batch identifier
        
    Returns:
        Conversation dictionary
    """
    # System prompt (can be empty or provide context)
    system_content = ""
    
    # User question (prompt for the model)
    user_content = f"""The following is a list of {n_genes} gene names ordered by descending expression level in a {organism} cell. Your task is to give the cell type which this cell belongs to based on its gene expression.
Cell sentence: {cell_sentence}.
The cell type corresponding to these genes is:"""
    
    # Assistant answer (ground truth)
    assistant_content = ground_truth
    
    conversation = {
        "conversations": [
            {
                "from": "system",
                "value": system_content
            },
            {
                "from": "human",
                "value": user_content
            },
            {
                "from": "gpt",
                "value": assistant_content
            }
        ],
        "dataset_id": dataset_id,
        "index": index,
        "group": group
    }
    
    return conversation


def convert_h5ad_to_jsonl(
    data_path: str,
    output_file: str,
    dataset_id: str = "unknown",
    n_genes: int = 200,
    organism: str = "Homo sapiens",
    seed: int = 1234
):
    """
    Convert h5ad file to conversations format JSONL.
    
    Args:
        data_path: Path to .h5ad file
        output_file: Path to output JSONL file
        dataset_id: Dataset identifier
        n_genes: Number of top genes to use
        organism: Organism name
        seed: Random seed
    """
    set_seed(seed)
    
    # Load and prepare data
    arrow_ds, vocabulary, adata = load_and_prepare_data(data_path, seed)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Convert to conversations format
    print(f"[INFO] Converting {len(arrow_ds)} cells to conversations format...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in tqdm(range(len(arrow_ds))):
            sample = arrow_ds[idx]
            
            # Get cell sentence (top k genes)
            gene_names = sample['cell_sentence'].split()[:n_genes]
            cell_sentence = " ".join(gene_names)
            
            # Get ground truth
            ground_truth = sample.get('cell_type', 'Unknown')
            
            # Get group/batch identifier if available
            group = sample.get('batch', '') if 'batch' in sample else ""
            
            # Create conversation item
            conversation = create_conversation_item(
                cell_sentence=cell_sentence,
                ground_truth=ground_truth,
                n_genes=n_genes,
                organism=organism,
                index=idx,
                dataset_id=dataset_id,
                group=group
            )
            
            # Write to JSONL
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    print(f"\n[SUCCESS] Converted {len(arrow_ds)} cells to: {output_file}")
    print(f"[INFO] Dataset ID: {dataset_id}")
    print(f"[INFO] Number of genes per cell: {n_genes}")
    print(f"[INFO] Organism: {organism}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5ad to conversations format JSONL for vLLM"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .h5ad file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="unknown",
        help="Dataset identifier (e.g., A013, D099)"
    )
    parser.add_argument(
        "--n_genes",
        type=int,
        default=200,
        help="Number of top genes to use (default: 200)"
    )
    parser.add_argument(
        "--organism",
        type=str,
        default="Homo sapiens",
        help="Organism name (default: 'Homo sapiens')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    convert_h5ad_to_jsonl(
        data_path=args.data_path,
        output_file=args.output_file,
        dataset_id=args.dataset_id,
        n_genes=args.n_genes,
        organism=args.organism,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

