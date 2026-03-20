#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perturbation Response Prediction Inference Script

This script allows you to use a fine-tuned Cell2Sentence model to predict
how gene expression changes in response to perturbations.

Usage:
    python predict_perturbation.py \
        --model_path ./output/.../checkpoint-XXX \
        --input_file test_samples.jsonl \
        --output_file predictions.jsonl \
        --num_genes 500
"""

import os
import sys
import json
import argparse
from typing import List, Dict

# Add parent directory to path for development mode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs


def load_test_samples(input_file: str) -> List[Dict]:
    """
    Load test samples from JSONL file.
    
    Args:
        input_file: Path to JSONL file containing test samples
    
    Returns:
        List of test sample dictionaries
    """
    print(f"Loading test samples from: {input_file}")
    samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                samples.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(samples)} test samples")
    return samples


def create_inference_prompt(
    sample: Dict,
    num_genes: int
) -> str:
    """
    Create inference prompt from a test sample.
    
    Args:
        sample: Test sample dictionary
        num_genes: Number of genes to use
    
    Returns:
        Formatted prompt string
    """
    gene_lists = sample.get('gene_lists', {})
    pre_genes = gene_lists.get('pre_perturbation_genes', [])
    
    metadata = sample.get('metadata', {})
    treatment = metadata.get('treatment', 'Unknown')
    cell_type = metadata.get('cell_type', 'Unknown')
    
    # Limit to num_genes
    control_genes = pre_genes[:num_genes]
    control_sentence = ' '.join(control_genes)
    
    # Format prompt
    prompt = f"""Given the following cell sentence of {len(control_genes)} expressed genes representing a {cell_type} cell's basal state, predict the cell sentence after applying the perturbation: {treatment}.
Control cell sentence: {control_sentence}.

Perturbed cell sentence:"""
    
    return prompt


def predict_perturbations(
    model_path: str,
    input_file: str,
    output_file: str,
    num_genes: int = 500,
    max_tokens: int = 2000,
    batch_size: int = 1,
):
    """
    Run perturbation predictions on test samples.
    
    Args:
        model_path: Path to fine-tuned model checkpoint
        input_file: Path to input JSONL file with test samples
        output_file: Path to save prediction results
        num_genes: Number of genes to use in input
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference (currently only supports 1)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test samples
    test_samples = load_test_samples(input_file)
    
    if len(test_samples) == 0:
        print("No test samples found. Exiting.")
        return
    
    # Load fine-tuned model
    print(f"Loading model from: {model_path}")
    
    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=os.path.dirname(model_path),
        save_name="inference_model"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        csmodel.save_path,
        cache_dir=os.path.join(csmodel.save_dir, ".cache"),
        trust_remote_code=True
    ).to(device)
    
    print("Model loaded successfully")
    
    # Run predictions
    results = []
    print(f"\nRunning predictions on {len(test_samples)} samples...")
    
    for idx, sample in enumerate(tqdm(test_samples)):
        # Create prompt
        prompt = create_inference_prompt(sample, num_genes)
        
        # Generate prediction
        try:
            prediction = csmodel.generate_from_prompt(
                model=model,
                prompt=prompt,
                max_num_tokens=max_tokens
            )
            
            # Remove trailing period if present
            if prediction.endswith('.'):
                prediction = prediction[:-1]
            
            # Get ground truth
            gene_lists = sample.get('gene_lists', {})
            post_genes = gene_lists.get('post_perturbation_genes', [])
            ground_truth = ' '.join(post_genes[:num_genes])
            
            # Prepare result
            result = {
                'sample_id': sample.get('id', f'sample_{idx}'),
                'cell_type': sample['metadata']['cell_type'],
                'treatment': sample['metadata']['treatment'],
                'prompt': prompt,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'metadata': sample['metadata']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # Save results
    print(f"\nSaving predictions to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} predictions")
    
    # Print a few sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    for i, result in enumerate(results[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Cell Type: {result['cell_type']}")
        print(f"Treatment: {result['treatment']}")
        print(f"\nControl genes (first 20):")
        control_genes = result['prompt'].split('Control cell sentence: ')[1].split('\n')[0].split()[:20]
        print(' '.join(control_genes))
        print(f"\nPredicted genes (first 20):")
        pred_genes = result['prediction'].split()[:20]
        print(' '.join(pred_genes))
        print(f"\nGround truth genes (first 20):")
        gt_genes = result['ground_truth'].split()[:20]
        print(' '.join(gt_genes))
        print("-" * 80)
    
    print("\nPrediction complete!")
    
    return results


def compute_gene_overlap_metrics(results: List[Dict], top_k: int = 50):
    """
    Compute gene overlap metrics between predictions and ground truth.
    
    Args:
        results: List of prediction result dictionaries
        top_k: Number of top genes to consider for overlap
    """
    print(f"\n{'=' * 80}")
    print(f"COMPUTING OVERLAP METRICS (Top {top_k} genes)")
    print(f"{'=' * 80}\n")
    
    overlaps = []
    precisions = []
    recalls = []
    
    for result in results:
        pred_genes = set(result['prediction'].split()[:top_k])
        gt_genes = set(result['ground_truth'].split()[:top_k])
        
        if len(gt_genes) == 0:
            continue
        
        overlap = len(pred_genes & gt_genes)
        precision = overlap / len(pred_genes) if len(pred_genes) > 0 else 0
        recall = overlap / len(gt_genes) if len(gt_genes) > 0 else 0
        
        overlaps.append(overlap)
        precisions.append(precision)
        recalls.append(recall)
    
    if len(overlaps) > 0:
        print(f"Average gene overlap: {sum(overlaps) / len(overlaps):.2f} / {top_k}")
        print(f"Average precision: {sum(precisions) / len(precisions):.4f}")
        print(f"Average recall: {sum(recalls) / len(recalls):.4f}")
        print(f"Average F1 score: {2 * sum(precisions) * sum(recalls) / (sum(precisions) + sum(recalls)) / len(precisions):.4f}")
    else:
        print("No valid samples for metrics computation")


def main():
    parser = argparse.ArgumentParser(
        description="Run perturbation response predictions with fine-tuned model"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file with test samples"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        help="Path to save prediction results"
    )
    parser.add_argument(
        "--num_genes",
        type=int,
        default=500,
        help="Number of genes to use in input and output"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only supports 1)"
    )
    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help="Compute overlap metrics after prediction"
    )
    parser.add_argument(
        "--top_k_overlap",
        type=int,
        default=50,
        help="Number of top genes to consider for overlap metrics"
    )
    
    args = parser.parse_args()
    
    # Run predictions
    results = predict_perturbations(
        model_path=args.model_path,
        input_file=args.input_file,
        output_file=args.output_file,
        num_genes=args.num_genes,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    
    # Compute metrics if requested
    if args.compute_metrics and results:
        compute_gene_overlap_metrics(results, top_k=args.top_k_overlap)


if __name__ == "__main__":
    main()

