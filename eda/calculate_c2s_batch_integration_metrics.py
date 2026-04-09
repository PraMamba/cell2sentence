#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell2Sentence Batch Integration Metrics Calculator (scIB Benchmark)

This tool calculates standard scIB (single-cell Integration Benchmark) metrics
to evaluate batch integration quality for Cell2Sentence embeddings, including:

1. Biological Variance Conservation:
   - NMI_cell (Normalized Mutual Information based on cell type)
   - ARI_cell (Adjusted Rand Index based on cell type)
   - ASW_cell (Average Silhouette Width based on cell type)

2. Batch Correction:
   - ASW_batch (Average Silhouette Width based on batch)
   - kNN Graph Connectivity

3. Aggregate Scores:
   - AvgBio (Average biological conservation score)
   - AvgBatch (Average batch correction score)
   - Overall_scIB (Overall scIB score)

Usage:
    python calculate_c2s_batch_integration_metrics.py \
        --data_path /path/to/dataset.h5ad \
        --model_path /path/to/c2s_model \
        --output_dir ./output \
        --batch_key dataset \
        --cell_type_key cell_type \
        --n_genes 200 \
        --k_neighbors 15 \
        --max_samples 10000
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import sys
import argparse
from pathlib import Path
import anndata

# Add cell2sentence to path
sys.path.insert(0, '/home/scbjtfy/cell2sentence/src')
import cell2sentence as cs

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set a consistent font for plots
plt.rcParams["font.family"] = "Times New Roman"


class C2SBatchIntegrationMetricsCalculator:
    """Calculator for batch integration metrics following scIB benchmark."""

    def __init__(self, data_path, model_path, output_dir,
                 batch_key='dataset', cell_type_key='cell_type',
                 n_genes=200, k_neighbors=15, max_samples=-1):
        """Initializes the calculator with configurable parameters."""

        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.batch_key = batch_key
        self.cell_type_key = cell_type_key
        self.n_genes = n_genes
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Model components
        self.tokenizer = None
        self.embeddings = None

        # Data storage
        self.adata = None
        self.arrow_ds = None
        self.vocabulary = None

        # Results storage
        self.metrics = {}

    def load_model_embeddings(self):
        """Loads the model's embedding layer."""
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model path does not exist: {self.model_path}")
            return False

        print("Loading model and tokenizer for embeddings...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Tokenizer loaded. Vocabulary size: {self.tokenizer.vocab_size}")

            print("Loading model embedding layer...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.embeddings = model.get_input_embeddings().weight.detach().cpu()
            print(f"Embedding matrix loaded. Shape: {self.embeddings.shape}")

            del model  # Free up memory
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_h5ad_data(self):
        """Loads h5ad data file and converts to cell sentences."""
        print(f"Loading h5ad file: {self.data_path}")
        try:
            self.adata = anndata.read_h5ad(self.data_path)
            print(f"Loaded {self.adata.n_obs} cells, {self.adata.n_vars} genes")
            print(f"Available metadata columns: {list(self.adata.obs.columns)}")

            # Check if required keys exist
            if self.batch_key not in self.adata.obs.columns:
                print(f"WARNING: Batch key '{self.batch_key}' not found in metadata")
                print(f"Available columns: {list(self.adata.obs.columns)}")
                # Try to find alternative
                for alt_key in ['dataset', 'batch', 'sample']:
                    if alt_key in self.adata.obs.columns:
                        print(f"Using '{alt_key}' as batch key instead")
                        self.batch_key = alt_key
                        break

            if self.cell_type_key not in self.adata.obs.columns:
                print(f"WARNING: Cell type key '{self.cell_type_key}' not found in metadata")
                # Try alternatives
                for alt_key in ['celltype', 'cell_ontology_class', 'free_annotation']:
                    if alt_key in self.adata.obs.columns:
                        print(f"Using '{alt_key}' as cell type key instead")
                        self.cell_type_key = alt_key
                        break

            # Sample data if max_samples is specified
            if self.max_samples > 0 and self.adata.n_obs > self.max_samples:
                print(f"Sampling {self.max_samples} cells from {self.adata.n_obs} total cells...")
                sample_indices = np.random.choice(
                    self.adata.n_obs,
                    size=self.max_samples,
                    replace=False
                )
                self.adata = self.adata[sample_indices, :].copy()
                print(f"Sampled dataset shape: {self.adata.shape}")

            # Convert to cell sentences
            print("Converting to Cell2Sentence format...")
            obs_cols = self.adata.obs.columns.tolist()
            self.arrow_ds, self.vocabulary = cs.CSData.adata_to_arrow(
                adata=self.adata,
                random_state=42,
                sentence_delimiter=' ',
                label_col_names=obs_cols
            )

            print(f"Created {len(self.arrow_ds)} cell sentences")
            print(f"Vocabulary size: {len(self.vocabulary)}")

            return True
        except Exception as e:
            print(f"ERROR: Failed to load h5ad file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_cell_embeddings(self):
        """Extracts embeddings for all cells using the C2S model."""
        print("Extracting cell embeddings...")

        embeddings_list = []
        batch_labels = []
        cell_type_labels = []

        for idx in tqdm(range(len(self.arrow_ds)), desc="Processing cells"):
            item = self.arrow_ds[idx]
            cell_sentence = item['sentence']

            # Truncate to top n_genes
            genes = cell_sentence.split()[:self.n_genes]
            truncated_sentence = ' '.join(genes)

            # Get embedding
            embedding = self.sentence_to_embedding(truncated_sentence)

            if embedding is not None:
                embeddings_list.append(embedding)

                # Extract batch label
                batch = item.get(self.batch_key, 'unknown')
                batch_labels.append(str(batch))

                # Extract cell type label
                cell_type = item.get(self.cell_type_key, 'unknown')
                cell_type_labels.append(str(cell_type))

        print(f"Extracted {len(embeddings_list)} valid embeddings")
        return np.array(embeddings_list), np.array(batch_labels), np.array(cell_type_labels)

    def sentence_to_embedding(self, cell_sentence):
        """Converts a cell sentence to its embedding vector (average pooling)."""
        try:
            # Tokenize the sentence
            tokens = self.tokenizer(
                cell_sentence,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=False
            )

            input_ids = tokens['input_ids'][0]  # Remove batch dimension

            # Get embeddings for each token
            token_embeddings = []
            for token_id in input_ids:
                if token_id < len(self.embeddings):
                    token_embeddings.append(self.embeddings[token_id].float())

            if not token_embeddings:
                return None

            # Average pooling across all tokens
            avg_embedding = torch.stack(token_embeddings).mean(dim=0)

            return avg_embedding.numpy()

        except Exception as e:
            return None

    def prepare_embeddings_and_labels(self):
        """Prepares embeddings and labels for metric calculation."""
        print("\n" + "="*80)
        print("Preparing Embeddings and Labels")
        print("="*80)

        # Extract embeddings
        embeddings, batch_labels, cell_type_labels = self.extract_cell_embeddings()

        if len(embeddings) == 0:
            print("ERROR: No valid embeddings found.")
            return None, None, None

        print(f"\nTotal embeddings: {len(embeddings)}")
        print(f"Embedding shape: {embeddings.shape}")

        print(f"\nLabel statistics:")
        print(f"  - Unique batches: {len(np.unique(batch_labels))}")
        print(f"  - Unique cell types: {len(np.unique(cell_type_labels))}")

        return embeddings, batch_labels, cell_type_labels

    def calculate_nmi_cell(self, embeddings, cell_type_labels):
        """Calculates Normalized Mutual Information based on cell type."""
        print("\nCalculating NMI_cell (Normalized Mutual Information)...")

        # Perform clustering (using optimal number of clusters = number of cell types)
        n_clusters = len(np.unique(cell_type_labels))
        print(f"  - Performing k-means clustering with {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate NMI
        nmi_score = normalized_mutual_info_score(cell_type_labels, cluster_labels)
        print(f"  - NMI_cell: {nmi_score:.4f}")

        return nmi_score

    def calculate_ari_cell(self, embeddings, cell_type_labels):
        """Calculates Adjusted Rand Index based on cell type."""
        print("\nCalculating ARI_cell (Adjusted Rand Index)...")

        # Perform clustering
        n_clusters = len(np.unique(cell_type_labels))
        print(f"  - Performing k-means clustering with {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate ARI
        ari_score = adjusted_rand_score(cell_type_labels, cluster_labels)
        print(f"  - ARI_cell: {ari_score:.4f}")

        return ari_score

    def calculate_asw_cell(self, embeddings, cell_type_labels):
        """Calculates Average Silhouette Width based on cell type."""
        print("\nCalculating ASW_cell (Average Silhouette Width based on cell type)...")

        # Encode cell type labels as integers
        le = LabelEncoder()
        cell_type_encoded = le.fit_transform(cell_type_labels)

        # Calculate silhouette scores
        print("  - Computing silhouette scores (this may take a while)...")

        # Use a sample if the dataset is too large
        if len(embeddings) > 10000:
            print(f"  - Dataset is large ({len(embeddings)} samples), using sample of 10000...")
            sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = cell_type_encoded[sample_indices]
            silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
        else:
            silhouette_avg = silhouette_score(embeddings, cell_type_encoded)

        # Normalize to [0, 1]
        asw_cell_final = (silhouette_avg + 1) / 2

        print(f"  - Raw ASW_cell: {silhouette_avg:.4f}")
        print(f"  - Normalized ASW_cell: {asw_cell_final:.4f}")

        return asw_cell_final

    def calculate_asw_batch(self, embeddings, batch_labels):
        """Calculates Average Silhouette Width based on batch (inverted)."""
        print("\nCalculating ASW_batch (Average Silhouette Width based on batch)...")

        # Encode batch labels as integers
        le = LabelEncoder()
        batch_encoded = le.fit_transform(batch_labels)

        # Calculate silhouette scores
        print("  - Computing silhouette scores (this may take a while)...")

        # Use a sample if the dataset is too large
        if len(embeddings) > 10000:
            print(f"  - Dataset is large ({len(embeddings)} samples), using sample of 10000...")
            sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = batch_encoded[sample_indices]
            silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
        else:
            silhouette_avg = silhouette_score(embeddings, batch_encoded)

        # Invert and normalize: lower raw ASW_batch is better (means batches are mixed)
        # Formula: 1 - (ASW_batch + 1) / 2
        asw_batch_final = 1 - (silhouette_avg + 1) / 2

        print(f"  - Raw ASW_batch: {silhouette_avg:.4f}")
        print(f"  - Inverted & Normalized ASW_batch: {asw_batch_final:.4f}")
        print(f"  - (Higher is better - indicates better batch mixing)")

        return asw_batch_final

    def calculate_knn_connectivity(self, embeddings, cell_type_labels, batch_labels):
        """Calculates kNN graph connectivity score."""
        print(f"\nCalculating kNN Graph Connectivity (k={self.k_neighbors})...")

        from sklearn.neighbors import kneighbors_graph

        # Build kNN graph
        print("  - Building kNN graph...")
        knn_graph = kneighbors_graph(
            embeddings,
            n_neighbors=self.k_neighbors,
            mode='connectivity',
            include_self=False
        )

        # For each cell type, check connectivity
        unique_cell_types = np.unique(cell_type_labels)
        connectivity_scores = []

        print(f"  - Checking connectivity for {len(unique_cell_types)} cell types...")

        for cell_type in unique_cell_types:
            # Get indices of cells of this type
            cell_type_mask = cell_type_labels == cell_type
            cell_type_indices = np.where(cell_type_mask)[0]

            if len(cell_type_indices) < 2:
                continue

            # Extract subgraph for this cell type
            subgraph = knn_graph[cell_type_indices][:, cell_type_indices]

            # Calculate number of connected components
            n_components, labels = connected_components(
                csgraph=subgraph,
                directed=False,
                return_labels=True
            )

            # Connectivity score: 1 means fully connected (only 1 component)
            # 0 means every cell is isolated (n components = n cells)
            if len(cell_type_indices) > 1:
                score = 1 - (n_components - 1) / (len(cell_type_indices) - 1)
            else:
                score = 1.0

            connectivity_scores.append(score)

        # Average connectivity across all cell types
        avg_connectivity = np.mean(connectivity_scores)

        print(f"  - kNN Connectivity: {avg_connectivity:.4f}")
        print(f"  - (Higher is better - indicates cell types remain connected after integration)")

        return avg_connectivity

    def calculate_aggregate_scores(self):
        """Calculates aggregate scIB scores."""
        print("\n" + "="*80)
        print("Calculating Aggregate Scores")
        print("="*80)

        # Extract individual metrics
        nmi_cell = self.metrics.get('NMI_cell', 0)
        ari_cell = self.metrics.get('ARI_cell', 0)
        asw_cell = self.metrics.get('ASW_cell_normalized', 0)
        asw_batch = self.metrics.get('ASW_batch_inverted_normalized', 0)
        knn_conn = self.metrics.get('kNN_connectivity', 0)

        # Calculate AvgBio (average biological conservation score)
        avg_bio = (nmi_cell + ari_cell + asw_cell) / 3

        # Calculate AvgBatch (average batch correction score)
        avg_batch = (asw_batch + knn_conn) / 2

        # Calculate Overall scIB score (weighted: 60% bio, 40% batch)
        overall_scib = 0.6 * avg_bio + 0.4 * avg_batch

        self.metrics['AvgBio'] = avg_bio
        self.metrics['AvgBatch'] = avg_batch
        self.metrics['Overall_scIB'] = overall_scib

        print(f"\nAggregate Scores:")
        print(f"  - AvgBio (Biological Conservation): {avg_bio:.4f}")
        print(f"  - AvgBatch (Batch Correction): {avg_batch:.4f}")
        print(f"  - Overall_scIB: {overall_scib:.4f}")

        return avg_bio, avg_batch, overall_scib

    def save_results(self):
        """Saves metrics to JSON and generates a summary report."""
        print("\n" + "="*80)
        print("Saving Results")
        print("="*80)

        # Save metrics to JSON
        json_path = os.path.join(self.output_dir, 'batch_integration_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to: {json_path}")

        # Generate summary report
        report_path = os.path.join(self.output_dir, 'batch_integration_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Cell2Sentence Batch Integration Metrics Report (scIB Benchmark)\n")
            f.write("="*80 + "\n\n")

            f.write("1. Biological Variance Conservation\n")
            f.write("-" * 40 + "\n")
            f.write(f"NMI_cell (Normalized Mutual Information):  {self.metrics.get('NMI_cell', 0):.4f}\n")
            f.write(f"ARI_cell (Adjusted Rand Index):            {self.metrics.get('ARI_cell', 0):.4f}\n")
            f.write(f"ASW_cell (Silhouette Width, normalized):   {self.metrics.get('ASW_cell_normalized', 0):.4f}\n")
            f.write(f"AvgBio (Average Bio Score):                {self.metrics.get('AvgBio', 0):.4f}\n\n")

            f.write("2. Batch Correction\n")
            f.write("-" * 40 + "\n")
            f.write(f"ASW_batch (Silhouette Width, inverted):    {self.metrics.get('ASW_batch_inverted_normalized', 0):.4f}\n")
            f.write(f"kNN Connectivity:                           {self.metrics.get('kNN_connectivity', 0):.4f}\n")
            f.write(f"AvgBatch (Average Batch Score):            {self.metrics.get('AvgBatch', 0):.4f}\n\n")

            f.write("3. Overall Score\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall_scIB (60% Bio + 40% Batch):        {self.metrics.get('Overall_scIB', 0):.4f}\n\n")

            f.write("="*80 + "\n")
            f.write("Interpretation Guide\n")
            f.write("="*80 + "\n")
            f.write("- All scores range from 0 to 1 (higher is better)\n")
            f.write("- NMI/ARI: Measures how well clustering matches true cell types\n")
            f.write("- ASW_cell: Measures separation between cell types\n")
            f.write("- ASW_batch: Measures batch mixing (inverted - high = well mixed)\n")
            f.write("- kNN Connectivity: Measures if cell types remain connected\n")
            f.write("- AvgBio: Should be high to preserve biological structure\n")
            f.write("- AvgBatch: Should be high to indicate good batch correction\n")
            f.write("- Overall_scIB: Balance between bio preservation and batch correction\n")

        print(f"Summary report saved to: {report_path}")

    def visualize_metrics(self):
        """Creates visualization of metrics."""
        print("\nGenerating metrics visualization...")

        # Prepare data for visualization
        bio_metrics = {
            'NMI_cell': self.metrics.get('NMI_cell', 0),
            'ARI_cell': self.metrics.get('ARI_cell', 0),
            'ASW_cell': self.metrics.get('ASW_cell_normalized', 0),
        }

        batch_metrics = {
            'ASW_batch': self.metrics.get('ASW_batch_inverted_normalized', 0),
            'kNN_conn': self.metrics.get('kNN_connectivity', 0),
        }

        aggregate_metrics = {
            'AvgBio': self.metrics.get('AvgBio', 0),
            'AvgBatch': self.metrics.get('AvgBatch', 0),
            'Overall': self.metrics.get('Overall_scIB', 0),
        }

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot biological conservation metrics
        ax1 = axes[0]
        bars1 = ax1.bar(bio_metrics.keys(), bio_metrics.values(),
                       color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.7)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Biological Variance Conservation', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # Plot batch correction metrics
        ax2 = axes[1]
        bars2 = ax2.bar(batch_metrics.keys(), batch_metrics.values(),
                       color=['#e74c3c', '#f39c12'], alpha=0.7)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Batch Correction', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # Plot aggregate metrics
        ax3 = axes[2]
        bars3 = ax3.bar(aggregate_metrics.keys(), aggregate_metrics.values(),
                       color=['#1abc9c', '#e67e22', '#c0392b'], alpha=0.7)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Aggregate Scores', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.legend()

        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, 'batch_integration_metrics.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Metrics visualization saved to: {fig_path}")

    def run_analysis(self):
        """Runs the complete batch integration metrics calculation."""
        print("="*80)
        print("Cell2Sentence Batch Integration Metrics Calculator (scIB Benchmark)")
        print("="*80)

        if not self.load_h5ad_data():
            print("Analysis aborted due to data loading failure.")
            return

        if not self.load_model_embeddings():
            print("Analysis aborted due to model loading failure.")
            return

        # Prepare embeddings and labels
        embeddings, batch_labels, cell_type_labels = self.prepare_embeddings_and_labels()

        if embeddings is None:
            print("Analysis aborted due to embedding preparation failure.")
            return

        # Calculate metrics
        print("\n" + "="*80)
        print("Calculating Batch Integration Metrics")
        print("="*80)

        # Biological variance conservation metrics
        self.metrics['NMI_cell'] = self.calculate_nmi_cell(embeddings, cell_type_labels)
        self.metrics['ARI_cell'] = self.calculate_ari_cell(embeddings, cell_type_labels)
        self.metrics['ASW_cell_normalized'] = self.calculate_asw_cell(embeddings, cell_type_labels)

        # Batch correction metrics
        self.metrics['ASW_batch_inverted_normalized'] = self.calculate_asw_batch(embeddings, batch_labels)
        self.metrics['kNN_connectivity'] = self.calculate_knn_connectivity(embeddings, cell_type_labels, batch_labels)

        # Aggregate scores
        self.calculate_aggregate_scores()

        # Save results
        self.save_results()

        # Visualize metrics
        self.visualize_metrics()

        print("\n" + "="*80)
        print("Analysis Finished!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cell2Sentence Batch Integration Metrics Calculator (scIB Benchmark)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to h5ad file containing single-cell data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Cell2Sentence model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output metrics')
    parser.add_argument('--batch_key', type=str, default='dataset',
                       help='Column name in adata.obs for batch information')
    parser.add_argument('--cell_type_key', type=str, default='cell_type',
                       help='Column name in adata.obs for cell type information')
    parser.add_argument('--n_genes', type=int, default=200,
                       help='Number of top genes to use per cell')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for kNN connectivity calculation')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='Maximum number of cells to analyze (-1 for all)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    calculator = C2SBatchIntegrationMetricsCalculator(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_key=args.batch_key,
        cell_type_key=args.cell_type_key,
        n_genes=args.n_genes,
        k_neighbors=args.k_neighbors,
        max_samples=args.max_samples
    )

    calculator.run_analysis()
