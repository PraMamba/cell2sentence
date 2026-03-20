#!/usr/bin/env python3
"""
Comprehensive evaluation script for Top-K gene prediction results (JSONL format)

This script combines three functionalities:
1. Clean gene predictions (remove invalid gene names)
2. Calculate metrics (Kendall's τ and IoU)
3. Analyze results (detailed statistics and case studies)

Adapted from RVQ-Alpha/eval/topk_genes/evaluate_topk_genes.py for JSONL input format.
"""

import json
import csv
import re
import argparse
import numpy as np
from collections import Counter
from typing import List, Set, Dict
from pathlib import Path


def parse_gene_list(text: str) -> List[str]:
    """
    Parse gene list from text
    
    Supported formats:
    1. Space-separated: "GENE1 GENE2 GENE3"
    2. Numbered list: "1. GENE1\n2. GENE2"
    3. Comma-separated: "GENE1, GENE2, GENE3"
    """
    if not text:
        return []

    # Method 1: Try numbered list format
    numbered_pattern = r'^\d+\.\s*([A-Za-z0-9\-_\.]+)'
    matches = re.findall(numbered_pattern, text, re.MULTILINE)
    if matches:
        return matches

    # Method 2: Space-separated format (most common for our use case)
    if ' ' in text and ',' not in text:
        genes = [g.strip() for g in text.split()]
        genes = [g for g in genes if g and not g.startswith('Based on')]
    else:
        # Method 3: Comma-separated format
        genes = [g.strip() for g in text.split(',')]
        genes = [g for g in genes if g and not g.startswith('Based on')]

    # Filter out content that doesn't look like gene names
    filtered_genes = []
    for gene in genes:
        # Remove markdown formatting symbols
        gene = gene.replace('*', '').replace('_', '').strip()
        
        # Remove trailing periods
        if gene.endswith('.'):
            gene = gene[:-1].strip()

        # Only keep strings that look like gene names
        if gene and re.match(r'^[A-Za-z0-9\-\.]+$', gene):
            filtered_genes.append(gene)

    return filtered_genes


def load_all_valid_gene_names(gene_list_path: str) -> Set[str]:
    """Load all valid gene names from CSV file (including housekeeping genes)"""
    valid_genes = set()

    with open(gene_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_name = row['Gene_Name'].strip()
            if gene_name:
                valid_genes.add(gene_name)

    print(f"✓ Loaded {len(valid_genes):,} valid gene names (including housekeeping)")
    return valid_genes


def load_valid_gene_names(gene_list_path: str) -> Set[str]:
    """Load valid gene name list from CSV file, excluding housekeeping genes"""
    valid_genes = set()
    housekeeping_count = 0

    with open(gene_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_name = row['Gene_Name'].strip()
            housekeeping_noise = row.get('Housekeeping_Noise', '').strip()
            
            # Skip housekeeping genes (Housekeeping_Noise == 'True')
            if housekeeping_noise == 'True':
                housekeeping_count += 1
                continue
            
            if gene_name:
                valid_genes.add(gene_name)

    print(f"✓ Loaded {len(valid_genes):,} valid gene names")
    print(f"✓ Excluded {housekeeping_count:,} housekeeping genes")
    return valid_genes


def clean_gene_list_for_output(genes: List[str], valid_genes: Set[str]) -> List[str]:
    """
    Clean gene list for output file: remove duplicates and invalid genes.
    Preserves housekeeping genes. Preserves the first occurrence of each gene.
    """
    cleaned = []
    seen_genes = set()
    
    for gene in genes:
        # Only add valid genes that haven't been seen before
        if gene in valid_genes and gene not in seen_genes:
            cleaned.append(gene)
            seen_genes.add(gene)
    
    return cleaned


def clean_gene_list_for_metrics(genes: List[str], valid_genes_no_housekeeping: Set[str]) -> List[str]:
    """
    Clean gene list for metrics calculation: remove duplicates, invalid genes, and housekeeping genes.
    Preserves the first occurrence of each gene.
    """
    cleaned = []
    seen_genes = set()
    
    for gene in genes:
        # Only add valid non-housekeeping genes that haven't been seen before
        if gene in valid_genes_no_housekeeping and gene not in seen_genes:
            cleaned.append(gene)
            seen_genes.add(gene)
    
    return cleaned


def format_gene_list(genes: List[str]) -> str:
    """Format gene list as space-separated string"""
    return ' '.join(genes)


def load_jsonl(input_file: str) -> List[Dict]:
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], output_file: str):
    """Save data to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def clean_predictions(
    input_file: str,
    output_file: str,
    gene_list_path: str
):
    """
    Clean gene predictions by removing invalid gene names
    
    Args:
        input_file: Path to input predictions JSONL file
        output_file: Path to output cleaned JSONL file
        gene_list_path: Path to CSV file with valid gene names
    """
    print(f"\nProcessing file: {input_file}")

    # Load all valid gene names (including housekeeping genes for output file)
    valid_genes_all = load_all_valid_gene_names(gene_list_path)

    # Load data from JSONL
    data = load_jsonl(input_file)

    total_samples = len(data)
    total_predicted_genes_before = 0
    total_predicted_genes_after = 0
    total_ground_truth_genes_before = 0
    total_ground_truth_genes_after = 0
    total_invalid_genes_removed = 0
    total_duplicates_removed = 0
    samples_modified = 0

    invalid_genes_counter = {}

    # Process each sample
    for record in data:
        # Clean predicted_answer
        predicted_text = record.get('predicted_answer', '')
        predicted_genes = parse_gene_list(predicted_text)
        total_predicted_genes_before += len(predicted_genes)
        
        # Separate valid and invalid genes
        valid_predicted_genes = [g for g in predicted_genes if g in valid_genes_all]
        invalid_predicted_genes = [g for g in predicted_genes if g not in valid_genes_all]
        total_invalid_genes_removed += len(invalid_predicted_genes)
        
        # Count duplicates in valid genes (before deduplication)
        unique_valid_genes_before = len(set(valid_predicted_genes))
        duplicates_in_valid = len(valid_predicted_genes) - unique_valid_genes_before
        total_duplicates_removed += duplicates_in_valid

        # Clean gene list for output (removes invalid genes and duplicates, preserving housekeeping)
        cleaned_predicted_genes = clean_gene_list_for_output(predicted_genes, valid_genes_all)
        total_predicted_genes_after += len(cleaned_predicted_genes)

        # Clean ground_truth
        ground_truth_text = record.get('ground_truth', '')
        ground_truth_genes = parse_gene_list(ground_truth_text)
        total_ground_truth_genes_before += len(ground_truth_genes)
        cleaned_ground_truth_genes = clean_gene_list_for_output(ground_truth_genes, valid_genes_all)
        total_ground_truth_genes_after += len(cleaned_ground_truth_genes)

        # Count removed invalid genes
        for gene in invalid_predicted_genes:
            invalid_genes_counter[gene] = invalid_genes_counter.get(gene, 0) + 1

        if len(predicted_genes) != len(cleaned_predicted_genes) or len(ground_truth_genes) != len(cleaned_ground_truth_genes):
            samples_modified += 1

        # Update predicted_answer and ground_truth
        record['predicted_answer'] = format_gene_list(cleaned_predicted_genes)
        record['ground_truth'] = format_gene_list(cleaned_ground_truth_genes)
        
        # Add cleaning statistics
        record['num_genes_before_cleaning'] = len(predicted_genes)
        record['num_genes_after_cleaning'] = len(cleaned_predicted_genes)
        record['num_genes_removed'] = len(predicted_genes) - len(cleaned_predicted_genes)

    # Save cleaned data
    save_jsonl(data, output_file)

    # Print statistics
    print("\n" + "=" * 80)
    print("Cleaning Statistics")
    print("=" * 80)
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Modified samples: {samples_modified:,} ({samples_modified/total_samples*100:.2f}%)")
    print(f"\nPredicted genes:")
    print(f"  Before cleaning: {total_predicted_genes_before:,}")
    print(f"  After cleaning: {total_predicted_genes_after:,}")
    print(f"\nGround truth genes:")
    print(f"  Before cleaning: {total_ground_truth_genes_before:,}")
    print(f"  After cleaning: {total_ground_truth_genes_after:,}")
    total_removed = total_invalid_genes_removed + total_duplicates_removed
    print(f"\nTotal removed genes: {total_removed:,}")
    print(f"  - Invalid genes removed: {total_invalid_genes_removed:,}")
    print(f"  - Duplicates removed (preserving first occurrence): {total_duplicates_removed:,}")
    print(f"  Note: Housekeeping genes are preserved in output file")
    if total_predicted_genes_before > 0:
        print(f"Predicted genes retention rate: {total_predicted_genes_after/total_predicted_genes_before*100:.2f}%")

    # Top-20 most frequently removed invalid gene names
    print("\n" + "=" * 80)
    print("Top-20 Most Frequently Removed Invalid Gene Names")
    print("=" * 80)

    sorted_invalid = sorted(invalid_genes_counter.items(), key=lambda x: x[1], reverse=True)
    for i, (gene, count) in enumerate(sorted_invalid[:20], 1):
        freq = count / total_samples * 100
        print(f"{i:2d}. {gene:30s}: {count:>5} times ({freq:>5.2f}%)")

    print(f"\n✓ Cleaned results saved to: {output_file}")
    print("=" * 80)


def calculate_kendall_tau(true_ranking: List[str], pred_ranking: List[str]) -> Dict:
    """
    Calculate Kendall's τ (Tau) for gene ranking
    
    Args:
        true_ranking: True gene ranking list (ordered by expression from high to low)
        pred_ranking: Predicted gene ranking list
    
    Returns:
        Dictionary containing:
        - tau: Kendall's τ value
        - n_c: Number of concordant pairs
        - n_d: Number of discordant pairs
        - n: Total number of genes
        - n_pairs: Total number of pairs
    """
    # Build rank dictionaries (gene -> rank)
    true_rank = {gene: i for i, gene in enumerate(true_ranking)}
    pred_rank = {gene: i for i, gene in enumerate(pred_ranking)}

    # Only consider genes that appear in both lists
    common_genes = set(true_ranking) & set(pred_ranking)

    if len(common_genes) < 2:
        return {
            "tau": 0.0,
            "n_c": 0,
            "n_d": 0,
            "n": len(common_genes),
            "n_pairs": 0
        }

    common_genes = list(common_genes)
    n = len(common_genes)

    # Calculate concordant and discordant pairs
    n_c = 0  # concordant pairs
    n_d = 0  # discordant pairs

    for i in range(n):
        for j in range(i + 1, n):
            gene_i = common_genes[i]
            gene_j = common_genes[j]

            # Relative order in true ranking
            true_order = true_rank[gene_i] < true_rank[gene_j]

            # Relative order in predicted ranking
            pred_order = pred_rank[gene_i] < pred_rank[gene_j]

            if true_order == pred_order:
                n_c += 1  # concordant pair
            else:
                n_d += 1  # discordant pair

    # Total number of pairs
    n_pairs = n * (n - 1) // 2

    # Calculate Kendall's τ
    tau = (n_c - n_d) / n_pairs if n_pairs > 0 else 0.0

    return {
        "tau": tau,
        "n_c": n_c,
        "n_d": n_d,
        "n": n,
        "n_pairs": n_pairs
    }


def calculate_iou(true_genes: List[str], pred_genes: List[str]) -> Dict:
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list
    
    Returns:
        Dictionary containing:
        - iou: IoU value
        - intersection: Size of intersection
        - union: Size of union
        - precision: Precision
        - recall: Recall
    """
    true_set = set(true_genes)
    pred_set = set(pred_genes)

    intersection = true_set & pred_set
    union = true_set | pred_set

    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0

    return {
        "iou": iou,
        "intersection": len(intersection),
        "union": len(union),
        "precision": precision,
        "recall": recall,
        "true_count": len(true_set),
        "pred_count": len(pred_set)
    }


def calculate_precision_recall_at_k(true_genes: List[str], pred_genes: List[str], k_values: List[int] = None) -> Dict:
    """
    Calculate Precision@K and Recall@K for multiple K values
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list (ordered by confidence/rank)
        k_values: List of K values to evaluate (default: [10, 20, 50, 100, 200, 500])
    
    Returns:
        Dictionary containing precision@k and recall@k for each K
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200, 500]
    
    true_set = set(true_genes)
    max_k = max(k_values) if k_values else len(pred_genes)
    pred_top_k = pred_genes[:max_k]
    
    results = {}
    for k_original in k_values:
        # Use actual number of genes if K exceeds available genes
        k_actual = min(k_original, len(pred_genes))
        
        pred_k_set = set(pred_top_k[:k_actual])
        intersection = true_set & pred_k_set
        
        # Precision@K: intersection / min(K, actual_predicted_genes)
        precision_k = len(intersection) / k_actual if k_actual > 0 else 0.0
        # Recall@K: intersection / true_genes (always relative to true set size)
        recall_k = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0
        
        # Use original K value as key for consistency
        results[f"precision@{k_original}"] = precision_k
        results[f"recall@{k_original}"] = recall_k
    
    return results


def calculate_f1_at_k(true_genes: List[str], pred_genes: List[str], k_values: List[int] = None) -> Dict:
    """
    Calculate F1-score@K for multiple K values
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list (ordered by confidence/rank)
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary containing f1@k for each K
    """
    precision_recall = calculate_precision_recall_at_k(true_genes, pred_genes, k_values)
    
    results = {}
    for k in (k_values or [10, 20, 50, 100, 200, 500]):
        precision_k = precision_recall.get(f"precision@{k}", 0.0)
        recall_k = precision_recall.get(f"recall@{k}", 0.0)
        
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0
        results[f"f1@{k}"] = f1_k
    
    return results


def calculate_overlap_coefficient(true_genes: List[str], pred_genes: List[str]) -> float:
    """
    Calculate Overlap Coefficient (intersection over minimum set size)
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list
    
    Returns:
        Overlap coefficient value
    """
    true_set = set(true_genes)
    pred_set = set(pred_genes)
    
    intersection = true_set & pred_set
    min_size = min(len(true_set), len(pred_set))
    
    return len(intersection) / min_size if min_size > 0 else 0.0


def calculate_average_precision(true_genes: List[str], pred_genes: List[str]) -> float:
    """
    Calculate Average Precision (AP) - area under Precision-Recall curve
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list (ordered by confidence/rank)
    
    Returns:
        Average Precision value
    """
    true_set = set(true_genes)
    if len(true_set) == 0:
        return 0.0
    
    # Calculate precision at each position
    precisions = []
    relevant_count = 0
    
    for i, gene in enumerate(pred_genes):
        if gene in true_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
    
    # Average Precision is the mean of precisions at relevant positions
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(true_set)


def calculate_map_at_k(true_genes: List[str], pred_genes: List[str], k_values: List[int] = None) -> Dict:
    """
    Calculate Mean Average Precision (mAP@K) for multiple K values
    
    mAP@K is the Average Precision calculated only on the top-K predicted genes.
    
    Args:
        true_genes: True gene list
        pred_genes: Predicted gene list (ordered by confidence/rank)
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary containing map@k for each K
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200, 500]
    
    true_set = set(true_genes)
    if len(true_set) == 0:
        return {f"map@{k}": 0.0 for k in k_values}
    
    results = {}
    for k in k_values:
        # Use actual number of genes if K exceeds available genes
        k_actual = min(k, len(pred_genes))
        pred_top_k = pred_genes[:k_actual]
        
        # Calculate precision at each position in top-K
        precisions = []
        relevant_count = 0
        
        for i, gene in enumerate(pred_top_k):
            if gene in true_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        # mAP@K is the mean of precisions at relevant positions in top-K
        if len(precisions) == 0:
            map_k = 0.0
        else:
            # Normalize by number of true genes (not all may be in top-K)
            # Use intersection size as denominator for mAP@K
            intersection_size = len(true_set & set(pred_top_k))
            if intersection_size > 0:
                map_k = sum(precisions) / intersection_size
            else:
                map_k = 0.0
        
        results[f"map@{k}"] = map_k
    
    return results


def calculate_ndcg_at_k(true_genes: List[str], pred_genes: List[str], k_values: List[int] = None) -> Dict:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K)
    
    Args:
        true_genes: True gene list (ordered by importance/expression)
        pred_genes: Predicted gene list (ordered by confidence/rank)
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary containing ndcg@k for each K
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200, 500]
    
    true_set = set(true_genes)
    true_rank = {gene: i for i, gene in enumerate(true_genes)}  # Lower rank = more important
    
    def dcg_at_k(genes_list: List[str], k: int) -> float:
        """Calculate DCG@K"""
        dcg = 0.0
        for i in range(min(k, len(genes_list))):
            gene = genes_list[i]
            if gene in true_set:
                # Relevance: 1 if in true set, 0 otherwise
                # Position: i+1 (1-indexed)
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)  # log2(i+2) because i is 0-indexed
        return dcg
    
    def idcg_at_k(k: int) -> float:
        """Calculate Ideal DCG@K (all relevant genes at top)"""
        idcg = 0.0
        for i in range(min(k, len(true_genes))):
            idcg += 1.0 / np.log2(i + 2)
        return idcg
    
    results = {}
    for k in k_values:
        dcg_k = dcg_at_k(pred_genes, k)
        idcg_k = idcg_at_k(k)
        ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0
        results[f"ndcg@{k}"] = ndcg_k
    
    return results


def evaluate_single_sample(record: Dict, valid_genes_no_housekeeping: Set[str], k_values: List[int] = None) -> Dict:
    """
    Evaluate a single sample with comprehensive metrics
    
    Args:
        record: Record containing ground_truth and predicted_answer
        valid_genes_no_housekeeping: Set of valid gene names excluding housekeeping genes
        k_values: List of K values for @K metrics (default: [10, 20, 50, 100, 200, 500])
    
    Returns:
        Dictionary containing all metrics
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200, 500]
    
    # Parse gene lists
    true_genes_raw = parse_gene_list(record.get('ground_truth', ''))
    pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
    
    # Clean gene lists for metrics calculation (exclude housekeeping genes)
    true_genes = clean_gene_list_for_metrics(true_genes_raw, valid_genes_no_housekeeping)
    pred_genes = clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping)

    # Calculate Kendall's τ
    tau_metrics = calculate_kendall_tau(true_genes, pred_genes)

    # Calculate IoU and basic precision/recall
    iou_metrics = calculate_iou(true_genes, pred_genes)
    
    # Calculate Precision@K and Recall@K
    precision_recall_at_k = calculate_precision_recall_at_k(true_genes, pred_genes, k_values)
    
    # Calculate F1@K
    f1_at_k = calculate_f1_at_k(true_genes, pred_genes, k_values)
    
    # Calculate Overlap Coefficient
    overlap_coef = calculate_overlap_coefficient(true_genes, pred_genes)
    
    # Calculate Average Precision (full AP)
    ap = calculate_average_precision(true_genes, pred_genes)
    
    # Calculate mAP@K (Mean Average Precision at K)
    map_at_k = calculate_map_at_k(true_genes, pred_genes, k_values)
    
    # Calculate nDCG@K
    ndcg_at_k = calculate_ndcg_at_k(true_genes, pred_genes, k_values)

    return {
        "dataset_id": record.get('id', ''),
        "index": record.get('id', ''),
        "true_gene_count": len(true_genes),
        "pred_gene_count": len(pred_genes),
        **tau_metrics,
        **iou_metrics,
        **precision_recall_at_k,
        **f1_at_k,
        "overlap_coefficient": overlap_coef,
        "average_precision": ap,
        **map_at_k,
        **ndcg_at_k
    }


def calculate_metrics(
    input_file: str,
    output_dir: str = None,
    gene_list_path: str = None
) -> Dict:
    """
    Calculate Kendall's τ and IoU metrics for all samples
    
    Args:
        input_file: Path to predictions JSONL file
        output_dir: Output directory for detailed results (optional)
        gene_list_path: Path to CSV file with valid gene names (optional, defaults to standard path)
    
    Returns:
        Dictionary containing summary statistics
    """
    # Load valid gene names excluding housekeeping genes for metrics calculation
    if gene_list_path is None:
        gene_list_path = '/home/scbjtfy/RVQ-Alpha/data_utils/gene_name_list_with_index.csv'
    valid_genes_no_housekeeping = load_valid_gene_names(gene_list_path)
    
    # Load data from JSONL
    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)

    print(f"Total records: {len(data)}")
    print(f"Note: Metrics are calculated excluding housekeeping genes")

    # Evaluate
    print("\nEvaluating...")
    k_values = [10, 20, 50, 100, 200, 500]
    results = []

    for record in data:
        try:
            result = evaluate_single_sample(record, valid_genes_no_housekeeping, k_values)
            results.append(result)
        except Exception as e:
            print(f"Error processing record {record.get('id', 'unknown')}: {e}")

    # Extract all metrics
    tau_values = [r['tau'] for r in results]
    iou_values = [r['iou'] for r in results]
    precision_values = [r['precision'] for r in results]
    recall_values = [r['recall'] for r in results]
    overlap_coef_values = [r['overlap_coefficient'] for r in results]
    ap_values = [r['average_precision'] for r in results]
    
    # Extract @K metrics
    precision_at_k_dict = {}
    recall_at_k_dict = {}
    f1_at_k_dict = {}
    map_at_k_dict = {}
    ndcg_at_k_dict = {}
    
    # For micro-averaging: aggregate intersections and totals across all samples
    precision_at_k_micro = {}
    recall_at_k_micro = {}
    
    for k in k_values:
        # Macro-averaging: collect individual precision/recall values
        precision_at_k_dict[k] = [r.get(f'precision@{k}', 0.0) for r in results]
        recall_at_k_dict[k] = [r.get(f'recall@{k}', 0.0) for r in results]
        f1_at_k_dict[k] = [r.get(f'f1@{k}', 0.0) for r in results]
        map_at_k_dict[k] = [r.get(f'map@{k}', 0.0) for r in results]
        ndcg_at_k_dict[k] = [r.get(f'ndcg@{k}', 0.0) for r in results]
        
        # Micro-averaging: aggregate intersections and totals
        # Calculate from precision/recall values to avoid re-parsing
        total_intersection = 0
        total_predicted_k = 0
        total_true = 0
        
        for r in results:
            true_count = r.get('true_gene_count', 0)
            pred_count = r.get('pred_gene_count', 0)
            k_actual = min(k, pred_count)
            
            # Get precision@k and recall@k for this sample
            precision_k_sample = r.get(f'precision@{k}', 0.0)
            recall_k_sample = r.get(f'recall@{k}', 0.0)
            
            # Calculate intersection from precision or recall
            # intersection = precision * k_actual = recall * true_count
            if k_actual > 0 and true_count > 0:
                # Use precision-based calculation (more accurate when k_actual is small)
                intersection = precision_k_sample * k_actual
                total_intersection += intersection
            elif true_count > 0:
                # Fallback to recall-based calculation
                intersection = recall_k_sample * true_count
                total_intersection += intersection
            
            total_predicted_k += k_actual
            total_true += true_count
        
        # Micro-averaged Precision@K and Recall@K
        precision_at_k_micro[k] = total_intersection / total_predicted_k if total_predicted_k > 0 else 0.0
        recall_at_k_micro[k] = total_intersection / total_true if total_true > 0 else 0.0

    # Calculate statistics
    stats = {
        "total_samples": len(results),
        "kendall_tau": {
            "mean": np.mean(tau_values),
            "std": np.std(tau_values),
            "median": np.median(tau_values),
            "min": np.min(tau_values),
            "max": np.max(tau_values),
            "values": tau_values
        },
        "iou": {
            "mean": np.mean(iou_values),
            "std": np.std(iou_values),
            "median": np.median(iou_values),
            "min": np.min(iou_values),
            "max": np.max(iou_values),
            "values": iou_values
        },
        "precision": {
            "mean": np.mean(precision_values),
            "std": np.std(precision_values),
            "median": np.median(precision_values)
        },
        "recall": {
            "mean": np.mean(recall_values),
            "std": np.std(recall_values),
            "median": np.median(recall_values)
        },
        "overlap_coefficient": {
            "mean": np.mean(overlap_coef_values),
            "std": np.std(overlap_coef_values),
            "median": np.median(overlap_coef_values)
        },
        "average_precision": {
            "mean": np.mean(ap_values),
            "std": np.std(ap_values),
            "median": np.median(ap_values)
        },
        "precision_at_k": {k: {
            "mean": np.mean(precision_at_k_dict[k]),  # Macro-averaged mean
            "std": np.std(precision_at_k_dict[k]),
            "median": np.median(precision_at_k_dict[k]),
            "micro": precision_at_k_micro[k]  # Micro-averaged (aggregated)
        } for k in k_values},
        "recall_at_k": {k: {
            "mean": np.mean(recall_at_k_dict[k]),  # Macro-averaged mean
            "std": np.std(recall_at_k_dict[k]),
            "median": np.median(recall_at_k_dict[k]),
            "micro": recall_at_k_micro[k]  # Micro-averaged (aggregated)
        } for k in k_values},
        "f1_at_k": {k: {
            "mean": np.mean(f1_at_k_dict[k]),
            "std": np.std(f1_at_k_dict[k]),
            "median": np.median(f1_at_k_dict[k])
        } for k in k_values},
        "map_at_k": {k: {
            "mean": np.mean(map_at_k_dict[k]),
            "std": np.std(map_at_k_dict[k]),
            "median": np.median(map_at_k_dict[k])
        } for k in k_values},
        "ndcg_at_k": {k: {
            "mean": np.mean(ndcg_at_k_dict[k]),
            "std": np.std(ndcg_at_k_dict[k]),
            "median": np.median(ndcg_at_k_dict[k])
        } for k in k_values},
        "detailed_results": results
    }

    # Print summary
    print_summary(stats)

    # Save detailed results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary statistics (without values list)
        stats_to_save = stats.copy()
        stats_to_save['kendall_tau'] = {k: v for k, v in stats['kendall_tau'].items() if k != 'values'}
        stats_to_save['iou'] = {k: v for k, v in stats['iou'].items() if k != 'values'}
        # Remove detailed_results from saved stats (too large)
        stats_to_save.pop('detailed_results', None)

        metrics_file = output_path / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"\nDetailed results saved to: {metrics_file}")

        # Save per-sample results
        per_sample_file = output_path / "per_sample_metrics.jsonl"
        save_jsonl(stats['detailed_results'], per_sample_file)
        print(f"Per-sample metrics saved to: {per_sample_file}")

    return stats


def print_summary(stats: Dict):
    """Print comprehensive evaluation summary"""
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"\nTotal Samples: {stats['total_samples']}")

    # Set-level Metrics
    print("\n" + "=" * 80)
    print("Set-level Metrics")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("IoU (Jaccard Index):")
    print("-" * 80)
    iou = stats['iou']
    print(f"  Mean:   {iou['mean']:.4f}")
    print(f"  Std:    {iou['std']:.4f}")
    print(f"  Median: {iou['median']:.4f}")
    print(f"  Min:    {iou['min']:.4f}")
    print(f"  Max:    {iou['max']:.4f}")

    print("\n" + "-" * 80)
    print("Precision & Recall:")
    print("-" * 80)
    print(f"  Precision (mean): {stats['precision']['mean']:.4f}")
    print(f"  Recall (mean):    {stats['recall']['mean']:.4f}")

    print("\n" + "-" * 80)
    print("Overlap Coefficient:")
    print("-" * 80)
    overlap = stats['overlap_coefficient']
    print(f"  Mean:   {overlap['mean']:.4f}")
    print(f"  Std:    {overlap['std']:.4f}")
    print(f"  Median: {overlap['median']:.4f}")

    # Precision@K and Recall@K
    print("\n" + "-" * 80)
    print("Precision@K & Recall@K (Macro-averaged):")
    print("-" * 80)
    print(f"{'K':<6} {'Precision@K':<15} {'Recall@K':<15} {'F1@K':<15}")
    print("-" * 80)
    k_values = sorted(stats['precision_at_k'].keys())
    for k in k_values:
        p_k = stats['precision_at_k'][k]['mean']
        r_k = stats['recall_at_k'][k]['mean']
        f1_k = stats['f1_at_k'][k]['mean']
        print(f"{k:<6} {p_k:<15.4f} {r_k:<15.4f} {f1_k:<15.4f}")
    
    # Micro-averaged Precision@K and Recall@K
    print("\n" + "-" * 80)
    print("Precision@K & Recall@K (Micro-averaged):")
    print("Note: Micro-averaging aggregates all samples before calculation")
    print("-" * 80)
    print(f"{'K':<6} {'Precision@K':<15} {'Recall@K':<15}")
    print("-" * 80)
    for k in k_values:
        p_k_micro = stats['precision_at_k'][k].get('micro', 0.0)
        r_k_micro = stats['recall_at_k'][k].get('micro', 0.0)
        print(f"{k:<6} {p_k_micro:<15.4f} {r_k_micro:<15.4f}")

    # Rank-based Metrics
    print("\n" + "=" * 80)
    print("Rank-based Metrics")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("Kendall's τ (Tau):")
    print("-" * 80)
    tau = stats['kendall_tau']
    print(f"  Mean:   {tau['mean']:.4f}")
    print(f"  Std:    {tau['std']:.4f}")
    print(f"  Median: {tau['median']:.4f}")
    print(f"  Min:    {tau['min']:.4f}")
    print(f"  Max:    {tau['max']:.4f}")

    print("\n" + "-" * 80)
    print("Average Precision (AP):")
    print("-" * 80)
    ap = stats['average_precision']
    print(f"  Mean:   {ap['mean']:.4f}")
    print(f"  Std:    {ap['std']:.4f}")
    print(f"  Median: {ap['median']:.4f}")

    print("\n" + "-" * 80)
    print("mAP@K (Mean Average Precision at K):")
    print("-" * 80)
    print(f"{'K':<6} {'mAP@K':<15}")
    print("-" * 80)
    for k in k_values:
        map_k = stats['map_at_k'][k]['mean']
        print(f"{k:<6} {map_k:<15.4f}")

    print("\n" + "-" * 80)
    print("nDCG@K (Normalized Discounted Cumulative Gain):")
    print("-" * 80)
    print(f"{'K':<6} {'nDCG@K':<15}")
    print("-" * 80)
    for k in k_values:
        ndcg_k = stats['ndcg_at_k'][k]['mean']
        print(f"{k:<6} {ndcg_k:<15.4f}")

    print("\n" + "=" * 80)


def analyze_results(input_file: str, gene_list_path: str = None):
    """
    Deep analysis of Top-K gene prediction results
    
    Args:
        input_file: Path to predictions JSONL file
        gene_list_path: Path to CSV file with valid gene names (optional, defaults to standard path)
    """
    print("=" * 80)
    print("Top-K Gene Prediction - Detailed Analysis Report")
    print("=" * 80)
    print(f"Data source: {input_file}\n")
    
    # Load valid gene names excluding housekeeping genes for analysis
    if gene_list_path is None:
        gene_list_path = '/home/scbjtfy/RVQ-Alpha/data_utils/gene_name_list_with_index.csv'
    valid_genes_no_housekeeping = load_valid_gene_names(gene_list_path)

    # Load data from JSONL
    data = load_jsonl(input_file)

    print(f"Total samples: {len(data):,}")
    print(f"Note: Analysis excludes housekeeping genes\n")

    # 1. Basic statistics
    print("=" * 80)
    print("1. Gene Count Statistics")
    print("=" * 80)

    true_counts = []
    pred_counts = []

    for record in data:
        true_genes_raw = parse_gene_list(record.get('ground_truth', ''))
        pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
        
        # Clean gene lists for analysis (exclude housekeeping genes)
        true_genes = clean_gene_list_for_metrics(true_genes_raw, valid_genes_no_housekeeping)
        pred_genes = clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping)

        true_counts.append(len(true_genes))
        pred_counts.append(len(pred_genes))

    print(f"\nGround Truth gene count:")
    print(f"  Mean: {np.mean(true_counts):.1f}")
    print(f"  Median: {np.median(true_counts):.0f}")
    print(f"  Min: {np.min(true_counts)}")
    print(f"  Max: {np.max(true_counts)}")

    print(f"\nPredicted gene count:")
    print(f"  Mean: {np.mean(pred_counts):.1f}")
    print(f"  Median: {np.median(pred_counts):.0f}")
    print(f"  Min: {np.min(pred_counts)}")
    print(f"  Max: {np.max(pred_counts)}")

    # 2. Overlap analysis
    print("\n" + "=" * 80)
    print("2. Gene Overlap Analysis")
    print("=" * 80)

    overlaps = []
    precisions = []
    recalls = []

    for record in data:
        true_genes_raw = parse_gene_list(record.get('ground_truth', ''))
        pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
        
        # Clean gene lists for analysis (exclude housekeeping genes)
        true_genes = set(clean_gene_list_for_metrics(true_genes_raw, valid_genes_no_housekeeping))
        pred_genes = set(clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping))

        intersection = len(true_genes & pred_genes)
        union = len(true_genes | pred_genes)

        if union > 0:
            iou = intersection / union
            overlaps.append(iou)

        if len(pred_genes) > 0:
            precision = intersection / len(pred_genes)
            precisions.append(precision)

        if len(true_genes) > 0:
            recall = intersection / len(true_genes)
            recalls.append(recall)

    print(f"\nIoU (Set Overlap):")
    print(f"  Mean: {np.mean(overlaps):.4f} ({np.mean(overlaps)*100:.2f}%)")
    print(f"  Median: {np.median(overlaps):.4f}")
    print(f"  Min: {np.min(overlaps):.4f}")
    print(f"  Max: {np.max(overlaps):.4f}")

    print(f"\nPrecision (Prediction Accuracy):")
    print(f"  Mean: {np.mean(precisions):.4f} ({np.mean(precisions)*100:.2f}%)")

    print(f"\nRecall:")
    print(f"  Mean: {np.mean(recalls):.4f} ({np.mean(recalls)*100:.2f}%)")

    # IoU distribution
    print(f"\nIoU Distribution:")
    bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist, _ = np.histogram(overlaps, bins=bins)

    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(overlaps) * 100
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {count:>4} ({pct:>5.2f}%)")

    # 3. Case studies
    print("\n" + "=" * 80)
    print("3. Case Study Analysis")
    print("=" * 80)

    # Find best and worst cases
    samples_with_metrics = []
    for record in data:
        true_genes_raw = parse_gene_list(record.get('ground_truth', ''))
        pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
        
        # Clean gene lists for analysis (exclude housekeeping genes)
        true_genes = clean_gene_list_for_metrics(true_genes_raw, valid_genes_no_housekeeping)
        pred_genes = clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping)

        true_set = set(true_genes)
        pred_set = set(pred_genes)

        intersection = len(true_set & pred_set)
        union = len(true_set | pred_set)
        iou = intersection / union if union > 0 else 0

        samples_with_metrics.append({
            'record': record,
            'iou': iou,
            'true_genes': true_genes,
            'pred_genes': pred_genes,
            'intersection': intersection
        })

    samples_with_metrics.sort(key=lambda x: x['iou'], reverse=True)

    # Top 3 best cases
    print("\nTop 3 Best Predictions (Highest IoU):")
    for i, sample in enumerate(samples_with_metrics[:3]):
        print(f"\nCase #{i+1}:")
        print(f"  IoU: {sample['iou']:.4f}")
        print(f"  True gene count: {len(sample['true_genes'])}")
        print(f"  Predicted gene count: {len(sample['pred_genes'])}")
        print(f"  Overlapping genes: {sample['intersection']}")
        print(f"  True top 10: {', '.join(sample['true_genes'][:10])}")
        print(f"  Predicted top 10: {', '.join(sample['pred_genes'][:10])}")

    # Top 3 worst cases
    print("\nTop 3 Worst Predictions (Lowest IoU):")
    for i, sample in enumerate(samples_with_metrics[-3:][::-1]):
        print(f"\nCase #{i+1}:")
        print(f"  IoU: {sample['iou']:.4f}")
        print(f"  True gene count: {len(sample['true_genes'])}")
        print(f"  Predicted gene count: {len(sample['pred_genes'])}")
        print(f"  Overlapping genes: {sample['intersection']}")
        print(f"  True top 10: {', '.join(sample['true_genes'][:10])}")
        print(f"  Predicted top 10: {', '.join(sample['pred_genes'][:10])}")

    # 4. Predicted gene frequency analysis
    print("\n" + "=" * 80)
    print("4. Predicted Gene Frequency Analysis")
    print("=" * 80)

    all_pred_genes = []
    for record in data:
        pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
        # Clean gene list for analysis (exclude housekeeping genes)
        pred_genes = clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping)
        all_pred_genes.extend(pred_genes)

    gene_freq = Counter(all_pred_genes)

    print(f"\nTotal unique predicted genes: {len(gene_freq)}")
    print(f"\nTop-20 Most Frequently Predicted Genes:")
    for gene, count in gene_freq.most_common(20):
        freq = count / len(data) * 100
        print(f"  {gene:15s}: {count:>5} times ({freq:>5.2f}%)")

    # 5. Kendall's τ analysis
    print("\n" + "=" * 80)
    print("5. Ranking Correlation Analysis (Kendall's τ)")
    print("=" * 80)

    # Calculate Kendall's τ for each sample
    tau_values = []
    for record in data:
        true_genes_raw = parse_gene_list(record.get('ground_truth', ''))
        pred_genes_raw = parse_gene_list(record.get('predicted_answer', ''))
        
        # Clean gene lists for analysis (exclude housekeeping genes)
        true_genes = clean_gene_list_for_metrics(true_genes_raw, valid_genes_no_housekeeping)
        pred_genes = clean_gene_list_for_metrics(pred_genes_raw, valid_genes_no_housekeeping)

        tau_result = calculate_kendall_tau(true_genes, pred_genes)
        tau_values.append(tau_result['tau'])

    print(f"\nKendall's τ Statistics:")
    print(f"  Mean: {np.mean(tau_values):.4f}")
    print(f"  Median: {np.median(tau_values):.4f}")
    print(f"  Std: {np.std(tau_values):.4f}")
    print(f"  Min: {np.min(tau_values):.4f}")
    print(f"  Max: {np.max(tau_values):.4f}")

    # τ distribution
    print(f"\nKendall's τ Distribution:")
    bins = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    hist, _ = np.histogram(tau_values, bins=bins)

    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(tau_values) * 100
        print(f"  {bins[i]:>5.2f} to {bins[i+1]:>5.2f}: {count:>4} ({pct:>5.2f}%)")

    # 6. Interpretation
    print("\n" + "=" * 80)
    print("6. Result Interpretation")
    print("=" * 80)

    avg_iou = np.mean(overlaps)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_tau = np.mean(tau_values)

    print(f"\nCore Metrics:")
    print(f"  IoU: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
    print(f"  Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"  Recall: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"  Kendall's τ: {avg_tau:.4f}")

    print(f"\nPerformance Rating:")
    if avg_iou > 0.5:
        grade = "Excellent ✅"
    elif avg_iou > 0.3:
        grade = "Good"
    elif avg_iou > 0.1:
        grade = "Fair"
    elif avg_iou > 0.05:
        grade = "Poor ⚠️"
    else:
        grade = "Very Poor ❌"

    print(f"  IoU: {grade}")

    print(f"\nDetailed Explanation:")
    print(f"  - Model correctly predicted {avg_iou*100:.2f}% of gene sets on average")
    print(f"  - Only {avg_precision*100:.2f}% of predicted genes are correct")
    print(f"  - Only {avg_recall*100:.2f}% of true genes were recalled")
    print(f"  - Ranking correlation τ={avg_tau:.4f}, close to random level (0.0)")

    if avg_iou < 0.02:
        print(f"\n⚠️ Conclusion:")
        print(f"  Model has learned almost no effective gene prediction capability.")
        print(f"  IoU={avg_iou:.4f} means predictions and true gene lists have almost no overlap.")
        print(f"  This confirms the data analysis conclusion: 99.82% of samples are isolated cases, model cannot generalize.")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation script for Top-K gene prediction results (JSONL format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean predictions only
  python evaluate_topk_genes.py --mode clean --input_file predictions.jsonl --output_file cleaned.jsonl

  # Calculate metrics only
  python evaluate_topk_genes.py --mode calculate --input_file cleaned.jsonl --output_dir metrics/

  # Analyze results only
  python evaluate_topk_genes.py --mode analyze --input_file cleaned.jsonl

  # Run all steps sequentially
  python evaluate_topk_genes.py --mode all --input_file predictions.jsonl --output_file cleaned.jsonl --output_dir metrics/
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['clean', 'calculate', 'analyze', 'all'],
        required=True,
        help='Execution mode: clean (clean predictions), calculate (calculate metrics), analyze (analyze results), or all (run all steps)'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input predictions JSONL file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file for cleaned results (required for clean mode)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for metrics (optional for calculate mode)'
    )
    parser.add_argument(
        '--gene_list',
        type=str,
        default='/home/scbjtfy/RVQ-Alpha/data_utils/gene_name_list_with_index.csv',
        help='Path to CSV file with valid gene names (for clean mode)'
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode in ['clean', 'all']:
        if not args.output_file:
            parser.error("--output_file is required for clean mode")

    # Execute based on mode
    if args.mode == 'clean':
        clean_predictions(args.input_file, args.output_file, args.gene_list)
    elif args.mode == 'calculate':
        calculate_metrics(args.input_file, args.output_dir, args.gene_list)
    elif args.mode == 'analyze':
        analyze_results(args.input_file, args.gene_list)
    elif args.mode == 'all':
        # Run all steps sequentially
        import tempfile
        import os

        # Step 1: Clean
        if not args.output_file:
            # Generate temporary output file name
            base_name = Path(args.input_file).stem
            output_dir = Path(args.input_file).parent
            args.output_file = str(output_dir / f"{base_name}_cleaned.jsonl")

        print("\n" + "=" * 80)
        print("Step 1: Cleaning Predictions")
        print("=" * 80)
        clean_predictions(args.input_file, args.output_file, args.gene_list)

        # Step 2: Calculate metrics
        print("\n" + "=" * 80)
        print("Step 2: Calculating Metrics")
        print("=" * 80)
        calculate_metrics(args.output_file, args.output_dir, args.gene_list)

        # Step 3: Analyze results
        print("\n" + "=" * 80)
        print("Step 3: Analyzing Results")
        print("=" * 80)
        analyze_results(args.output_file, args.gene_list)


if __name__ == "__main__":
    main()

