# Cell2Sentence Embedding Analysis Tools

This directory contains tools for analyzing Cell2Sentence model embeddings and evaluating batch integration quality.

## Overview

Two main analysis tools are provided:

1. **Embedding Distribution Analysis** - Visualizes the embedding space using t-SNE
2. **Batch Integration Metrics** - Calculates scIB benchmark metrics for batch correction evaluation

## Prerequisites

- Python 3.7+
- Required packages: `anndata`, `scanpy`, `torch`, `transformers`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `numpy`
- A trained Cell2Sentence model
- Single-cell data in h5ad format

## Installation

```bash
# Install required packages
pip install anndata scanpy torch transformers scikit-learn matplotlib seaborn pandas numpy tqdm
```

## Usage

### 1. Embedding Distribution Analysis

This tool visualizes how cell embeddings are distributed in the latent space.

**Command Line:**
```bash
python analyze_c2s_embedding_distribution.py \
    --data_path /path/to/dataset.h5ad \
    --model_path /path/to/cell2sentence/model \
    --output_dir ./output \
    --n_genes 200 \
    --top_cell_types 20 \
    --max_samples 10000
```

**Using Shell Script:**
```bash
# Edit run_embedding_analysis.sh to set your paths
vim run_embedding_analysis.sh

# Run the analysis
./run_embedding_analysis.sh
```

**Parameters:**
- `--data_path`: Path to h5ad file containing single-cell data
- `--model_path`: Path to Cell2Sentence model checkpoint
- `--output_dir`: Directory to save output visualizations
- `--n_genes`: Number of top expressed genes to use per cell (default: 200)
- `--top_cell_types`: Number of top cell types to visualize, -1 for all (default: -1)
- `--max_samples`: Maximum number of cells to analyze, -1 for all (default: -1)

**Output Files:**
- `tsne_results.csv` - t-SNE coordinates and metadata for all cells
- `embedding_by_cell_type.png` - t-SNE visualization colored by cell type
- `embedding_by_tissue.png` - t-SNE visualization colored by tissue (if available)
- `embedding_by_batch.png` - t-SNE visualization colored by batch (if available)
- `embedding_by_dataset.png` - t-SNE visualization colored by dataset (if available)

### 2. Batch Integration Metrics Calculation

This tool calculates scIB (single-cell Integration Benchmark) metrics to evaluate batch integration quality.

**Command Line:**
```bash
python calculate_c2s_batch_integration_metrics.py \
    --data_path /path/to/dataset.h5ad \
    --model_path /path/to/cell2sentence/model \
    --output_dir ./output \
    --batch_key dataset \
    --cell_type_key cell_type \
    --n_genes 200 \
    --k_neighbors 15 \
    --max_samples 10000
```

**Using Shell Script:**
```bash
# Edit run_batch_integration_metrics.sh to set your paths
vim run_batch_integration_metrics.sh

# Run the analysis
./run_batch_integration_metrics.sh
```

**Parameters:**
- `--data_path`: Path to h5ad file containing single-cell data
- `--model_path`: Path to Cell2Sentence model checkpoint
- `--output_dir`: Directory to save output metrics
- `--batch_key`: Column name in adata.obs for batch information (default: 'dataset')
- `--cell_type_key`: Column name in adata.obs for cell type information (default: 'cell_type')
- `--n_genes`: Number of top expressed genes to use per cell (default: 200)
- `--k_neighbors`: Number of neighbors for kNN connectivity calculation (default: 15)
- `--max_samples`: Maximum number of cells to analyze, -1 for all (default: -1)

**Output Files:**
- `batch_integration_metrics.json` - All calculated metrics in JSON format
- `batch_integration_report.txt` - Human-readable summary report
- `batch_integration_metrics.png` - Bar chart visualization of all metrics

**Metrics Calculated:**

1. **Biological Variance Conservation:**
   - NMI_cell: Normalized Mutual Information (clustering vs true cell types)
   - ARI_cell: Adjusted Rand Index (clustering vs true cell types)
   - ASW_cell: Average Silhouette Width based on cell type

2. **Batch Correction:**
   - ASW_batch: Average Silhouette Width based on batch (inverted - higher is better)
   - kNN_connectivity: How well cell types remain connected after integration

3. **Aggregate Scores:**
   - AvgBio: Average biological conservation score
   - AvgBatch: Average batch correction score
   - Overall_scIB: Overall score (60% bio + 40% batch)

**Interpretation:**
- All scores range from 0 to 1, where higher is better
- Good integration should have high AvgBio (>0.7) to preserve biological structure
- Good integration should have high AvgBatch (>0.7) indicating effective batch mixing
- Overall_scIB balances both aspects

## Example Workflow

```bash
# 1. Create output directories
mkdir -p results/embedding_analysis results/batch_integration

# 2. Run embedding analysis
python analyze_c2s_embedding_distribution.py \
    --data_path /data/my_dataset.h5ad \
    --model_path /models/c2s_checkpoint \
    --output_dir results/embedding_analysis \
    --n_genes 200 \
    --top_cell_types 20 \
    --max_samples 5000

# 3. Run batch integration metrics
python calculate_c2s_batch_integration_metrics.py \
    --data_path /data/my_dataset.h5ad \
    --model_path /models/c2s_checkpoint \
    --output_dir results/batch_integration \
    --batch_key dataset \
    --cell_type_key cell_type \
    --n_genes 200 \
    --k_neighbors 15 \
    --max_samples 5000

# 4. Check results
ls -l results/embedding_analysis/
ls -l results/batch_integration/
```

## Data Format Requirements

Your h5ad file should contain:
- Gene expression data in `.X`
- Cell metadata in `.obs` with at least:
  - Cell type information (column name specified by `--cell_type_key`)
  - Batch/dataset information (column name specified by `--batch_key`)
  - Optional: tissue, donor, or other metadata fields

## Performance Considerations

- **Memory:** Loading large models and datasets requires substantial RAM
- **Computation Time:**
  - t-SNE can be slow for >10,000 cells
  - Silhouette score calculation is O(n²) and may be slow for large datasets
  - Use `--max_samples` to limit analysis to a subset of cells for faster results

- **Sampling Strategy:**
  - For large datasets (>50,000 cells), consider using `--max_samples 10000`
  - The tools will automatically sample if silhouette calculation exceeds 10,000 samples
  - Random sampling is used with a fixed seed for reproducibility

## Comparison with RVQ-Alpha Analysis

These tools are adapted from similar RVQ (Residual Vector Quantization) embedding analysis tools but modified for Cell2Sentence:

**Key Differences:**
- **Input:** h5ad files instead of CSV + parquet files
- **Embeddings:** Cell sentences converted to token embeddings (average pooling) instead of RVQ code embeddings
- **Data Processing:** Uses Cell2Sentence's `adata_to_arrow()` conversion
- **Model:** Standard Huggingface transformers instead of custom RVQ token embeddings

**Similarities:**
- Same scIB metrics (NMI, ARI, ASW, kNN connectivity)
- Same t-SNE visualization approach
- Same aggregate scoring methodology (60% bio + 40% batch)

## Troubleshooting

**Issue: "Column not found in metadata"**
- Check available columns in your h5ad file: `import anndata; adata = anndata.read_h5ad('data.h5ad'); print(adata.obs.columns)`
- Adjust `--batch_key` and `--cell_type_key` parameters accordingly

**Issue: "Out of memory"**
- Reduce `--max_samples` to analyze fewer cells
- Use a machine with more RAM
- Close other applications

**Issue: "t-SNE is very slow"**
- Reduce `--max_samples`
- Consider using UMAP instead (would require code modification)

**Issue: "Module not found: cell2sentence"**
- Ensure the script has the correct path to cell2sentence: `sys.path.insert(0, '/home/scbjtfy/cell2sentence/src')`
- Or install cell2sentence: `pip install -e /home/scbjtfy/cell2sentence`

## Citation

If you use these tools in your research, please cite:

- Cell2Sentence: [Original paper citation]
- scIB benchmark: Luecken et al., "Benchmarking atlas-level data integration in single-cell genomics", Nature Methods 2022

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.
