"""
Cell2Sentence Cell Type Prediction Script

This script extracts the prediction logic from the Cell2Sentence tutorial notebook
and provides a clean interface for zero-shot cell type prediction.

Usage:
    python c2s_predict.py \
        --data_path /path/to/dataset.h5ad \
        --model_path /path/to/C2S-Pythia-410m-cell-type-prediction \
        --output_dir /path/to/save/results \
        --dataset_id A013 \
        --n_genes 200
"""

import os
import random
import argparse
from typing import List, Dict
from datetime import datetime

import numpy as np
import anndata
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import C2SPromptFormatter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM


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


def create_csdata(arrow_ds, vocabulary, save_dir: str, save_name: str):
    """
    Create CSData object from arrow dataset.
    
    Args:
        arrow_ds: Arrow dataset
        vocabulary: Vocabulary dict
        save_dir: Directory to save CSData
        save_name: Name for the CSData
        
    Returns:
        CSData object
    """
    print(f"[INFO] Creating CSData object at: {save_dir}/{save_name}")
    
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds, 
        vocabulary=vocabulary,
        save_dir=save_dir,
        save_name=save_name,
        dataset_backend="arrow"
    )
    
    print(f"[INFO] CSData created: {csdata}")
    return csdata


def load_c2s_model(model_path: str, save_dir: str, save_name: str):
    """
    Load Cell2Sentence cell type prediction model.
    
    Args:
        model_path: Path to pretrained C2S model
        save_dir: Directory for model saving
        save_name: Name for the model
        
    Returns:
        CSModel object
    """
    print(f"[INFO] Loading Cell2Sentence model from: {model_path}")
    
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=save_dir,
        save_name=save_name
    )
    
    print(f"[INFO] Model loaded successfully")
    return csmodel


def predict_cell_types(csdata, csmodel, n_genes: int = 200):
    """
    Predict cell types using Cell2Sentence model and return model inputs.
    
    Args:
        csdata: CSData object
        csmodel: CSModel object
        n_genes: Number of top genes to use for prediction
        
    Returns:
        Tuple of (predicted_cell_types, model_inputs)
        - predicted_cell_types: List of predicted cell type names
        - model_inputs: List of actual model input strings
    """
    print(f"[INFO] Predicting cell types using top {n_genes} genes...")
    
    # Load model
    import os
    print("Reloading model from path on disk:", csmodel.save_path)
    model = AutoModelForCausalLM.from_pretrained(
        csmodel.save_path,
        cache_dir=os.path.join(csmodel.save_dir, ".cache"),
        trust_remote_code=True
    )
    model = model.to(csmodel.device)
    
    # Load data from csdata object
    hf_ds_dict = load_from_disk(csdata.data_path)
    
    # Format prompts
    prompt_formatter = C2SPromptFormatter(task="cell_type_prediction", top_k_genes=n_genes)
    formatted_hf_ds = prompt_formatter.format_hf_ds(hf_ds_dict)
    
    # Predict cell types and collect model inputs
    print(f"[INFO] Predicting cell types for {formatted_hf_ds.num_rows} cells using CSModel...")
    predicted_cell_types = []
    model_inputs = []
    
    for sample_idx in tqdm(range(formatted_hf_ds.num_rows)):
        # Get model input
        sample = formatted_hf_ds[sample_idx]
        model_input_prompt_str = sample["model_input"]
        model_inputs.append(model_input_prompt_str)
        
        # Generate prediction
        pred = csmodel.generate_from_prompt(model, prompt=model_input_prompt_str)
        predicted_cell_types.append(pred)
    
    print(f"[INFO] Predicted {len(predicted_cell_types)} cell types")
    
    # Clean predictions (remove trailing periods)
    cleaned_predictions = []
    for pred in predicted_cell_types:
        if pred and pred.endswith('.'):
            pred = pred[:-1]
        cleaned_predictions.append(pred)
    
    return cleaned_predictions, model_inputs


def create_qa_format(
    arrow_ds,
    adata,
    predictions: List[str],
    model_inputs: List[str],
    dataset_id: str
) -> List[Dict]:
    """
    Create QA format results compatible with standardized output.
    
    Args:
        arrow_ds: Arrow dataset
        adata: AnnData object
        predictions: List of predicted cell types
        model_inputs: List of actual model input strings
        dataset_id: Dataset identifier
        
    Returns:
        List of result dictionaries in standardized format
    """
    print("[INFO] Creating QA format results...")
    
    results = []
    
    for idx in range(len(arrow_ds)):
        # Get ground truth
        ground_truth = arrow_ds[idx].get('cell_type', 'Unknown')
        
        # Get predicted answer (already cleaned, no trailing period)
        predicted_answer = predictions[idx] if idx < len(predictions) else ""
        
        # Get actual model input (the question field)
        question = model_inputs[idx] if idx < len(model_inputs) else ""
        
        # Get actual model output (the full_response field)
        # The predictions from model are already cleaned (no trailing period)
        full_response = predicted_answer
        
        # Create result in standardized format
        result = {
            "model_name": "vandijklab/C2S-Pythia-410m-cell-type-prediction",
            "dataset_id": dataset_id,
            "index": idx,
            "task_type": "cell type",
            "task_variant": "singlecell_openended",
            "question": question,  # Actual model input
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "full_response": full_response,  # Actual model output (no <answer> tags)
            "group": ""
        }
        
        results.append(result)
    
    return results


def run_prediction(
    data_path: str,
    model_path: str,
    output_dir: str,
    dataset_id: str = "unknown",
    n_genes: int = 200,
    seed: int = 1234
):
    """
    Run complete prediction pipeline.
    
    Args:
        data_path: Path to .h5ad file
        model_path: Path to C2S model
        output_dir: Directory to save results
        dataset_id: Dataset identifier
        n_genes: Number of top genes for prediction
        seed: Random seed
    """
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and prepare data
    arrow_ds, vocabulary, adata = load_and_prepare_data(data_path, seed)
    
    # Create temporary directory for CSData
    csdata_dir = os.path.join(output_dir, "temp_csdata")
    os.makedirs(csdata_dir, exist_ok=True)
    
    # Create CSData
    csdata = create_csdata(
        arrow_ds,
        vocabulary,
        save_dir=csdata_dir,
        save_name=f"{dataset_id}_csdata"
    )
    
    # Load model
    model_save_dir = os.path.join(output_dir, "temp_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    csmodel = load_c2s_model(
        model_path,
        save_dir=model_save_dir,
        save_name=f"{dataset_id}_model"
    )
    
    # Predict and get model inputs
    predictions, model_inputs = predict_cell_types(csdata, csmodel, n_genes)
    
    # Create QA format results
    results = create_qa_format(arrow_ds, adata, predictions, model_inputs, dataset_id)
    
    # Save results
    output_file = os.path.join(
        output_dir,
        f"singlecell_openended_predictions_{timestamp}.json"
    )
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Predictions saved to: {output_file}")
    print(f"[INFO] Total predictions: {len(results)}")
    
    # Clean up temporary directories
    import shutil
    try:
        shutil.rmtree(csdata_dir)
        shutil.rmtree(model_save_dir)
    except:
        pass
    
    return results, output_file


def main():
    parser = argparse.ArgumentParser(
        description="Cell2Sentence Cell Type Prediction"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .h5ad file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Pythia-410m-cell-type-prediction/snapshots/5a4dc3b949b5868ca63752b37bc22e3b0216e435",
        help="Path to C2S model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
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
        help="Number of top genes for prediction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    run_prediction(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        n_genes=args.n_genes,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

