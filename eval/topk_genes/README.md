# Top-K Gene Prediction Evaluation

This directory contains evaluation scripts for Top-K gene prediction results, adapted from `RVQ-Alpha/eval/topk_genes/evaluate_topk_genes.py` for JSONL input format.

## Overview

The evaluation script provides comprehensive metrics for Top-K gene prediction tasks:

1. **Cleaning**: Remove invalid gene names and duplicates
2. **Metrics Calculation**: Calculate various evaluation metrics
3. **Analysis**: Detailed statistics and case studies

## Metrics

### Set-level Metrics
- **IoU (Jaccard Index)**: Intersection over Union
- **Precision & Recall**: Basic precision and recall
- **Overlap Coefficient**: Intersection over minimum set size

### Top-K Metrics
- **Precision@K**: Precision at top K predictions (K = 10, 20, 50, 100, 200, 500)
- **Recall@K**: Recall at top K predictions
- **F1@K**: F1-score at top K predictions
- **mAP@K**: Mean Average Precision at K
- **nDCG@K**: Normalized Discounted Cumulative Gain at K

### Rank-based Metrics
- **Kendall's τ**: Ranking correlation coefficient
- **Average Precision (AP)**: Area under Precision-Recall curve

## Usage

### Quick Start

Run the complete evaluation pipeline:

```bash
bash run_evaluate.sh
```

### Manual Usage

#### 1. Clean predictions only

```bash
python evaluate_topk_genes.py \
    --mode clean \
    --input_file predictions.jsonl \
    --output_file cleaned.jsonl \
    --gene_list /path/to/gene_name_list_with_index.csv
```

#### 2. Calculate metrics only

```bash
python evaluate_topk_genes.py \
    --mode calculate \
    --input_file cleaned.jsonl \
    --output_dir metrics/ \
    --gene_list /path/to/gene_name_list_with_index.csv
```

#### 3. Analyze results only

```bash
python evaluate_topk_genes.py \
    --mode analyze \
    --input_file cleaned.jsonl \
    --gene_list /path/to/gene_name_list_with_index.csv
```

#### 4. Run all steps

```bash
python evaluate_topk_genes.py \
    --mode all \
    --input_file predictions.jsonl \
    --output_file cleaned.jsonl \
    --output_dir metrics/ \
    --gene_list /path/to/gene_name_list_with_index.csv
```

## Input Format

The input file should be a JSONL file (one JSON object per line) with the following structure:

```json
{
  "id": "sample_id",
  "ground_truth": "GENE1 GENE2 GENE3 ...",
  "predicted_answer": "GENE1 GENE2 GENE3 ...",
  ...
}
```

The `ground_truth` and `predicted_answer` fields should contain space-separated gene names.

## Output Files

When running with `--output_dir`, the following files are generated:

- `evaluation_metrics.json`: Summary statistics (mean, std, median for all metrics)
- `per_sample_metrics.jsonl`: Per-sample detailed metrics

## Gene List Format

The gene list CSV file should have the following columns:
- `Gene_Name`: Gene name
- `Housekeeping_Noise`: "True" if housekeeping gene, otherwise empty or "False"

## Notes

- Housekeeping genes are excluded from metrics calculation but preserved in cleaned output files
- Invalid gene names are removed during cleaning
- Duplicate genes are removed (preserving first occurrence)
- All metrics are calculated excluding housekeeping genes

