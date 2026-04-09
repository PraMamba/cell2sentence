#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell2Sentence Embedding Distribution Analysis Tool

This tool analyzes Cell2Sentence embedding distributions based on h5ad data files.
It converts cell gene expression data to cell sentences, extracts embeddings from
the C2S model, and visualizes the distribution using t-SNE.

Usage:
    python analyze_c2s_embedding_distribution.py \
        --data_path /path/to/dataset.h5ad \
        --model_path /path/to/c2s_model \
        --output_dir ./output \
        --n_genes 200 \
        --top_cell_types 20 \
        --max_samples 10000
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
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


class C2SEmbeddingDistributionAnalyzer:
    """Analyzer for Cell2Sentence embedding distributions."""

    def __init__(self, data_path, model_path, output_dir,
                 n_genes=200, top_cell_types=-1, max_samples=-1):
        """Initializes the analyzer with configurable parameters."""

        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.n_genes = n_genes
        self.top_cell_types = top_cell_types
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

        # Visualization configuration
        self._setup_visualization_config()

    def _setup_visualization_config(self):
        """Sets up consistent visualization configuration."""
        # Color palettes for metadata fields
        self.color_palettes = {
            'cell_type': 'tab20',
            'dataset': 'Set3',
            'tissue': 'Paired',
            'batch': 'Set2'
        }

        # Master color mapping that will be consistent across all plots
        self.master_colors = {}

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
        labels_list = []

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

                # Extract labels
                label_info = {
                    'cell_name': item.get('cell_name', f'cell_{idx}'),
                }

                # Add all available metadata
                for col in self.adata.obs.columns:
                    if col in item:
                        label_info[col] = str(item[col])

                # Ensure we have at least cell_type
                if 'cell_type' not in label_info:
                    # Try alternative column names
                    for col_name in ['celltype', 'cell_ontology_class', 'free_annotation']:
                        if col_name in item:
                            label_info['cell_type'] = str(item[col_name])
                            break
                    if 'cell_type' not in label_info:
                        label_info['cell_type'] = 'unknown'

                labels_list.append(label_info)

        print(f"Extracted {len(embeddings_list)} valid embeddings")
        return np.array(embeddings_list), labels_list

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
            print(f"WARNING: Failed to get embedding: {e}")
            return None

    def _build_color_mappings(self, labels):
        """Builds color mappings for metadata fields."""
        print("Building color mappings for metadata fields...")

        # Collect all unique values for common fields
        all_field_values = {
            'cell_type': set(),
            'tissue': set(),
            'batch': set(),
            'dataset': set()
        }

        for label in labels:
            for field in all_field_values:
                if field in label:
                    value = label[field]
                    if value and value != 'unknown' and isinstance(value, str):
                        all_field_values[field].add(value)

        # Build color mapping for each field
        for field_name, unique_values in all_field_values.items():
            if not unique_values:
                continue

            self.master_colors[field_name] = {}
            sorted_values = sorted(unique_values)

            # Get the color palette for this field
            palette_name = self.color_palettes.get(field_name, 'tab20')

            if len(sorted_values) <= 20:
                colors = plt.cm.get_cmap(palette_name)(np.linspace(0, 1, len(sorted_values)))
            else:
                base_colors = plt.cm.get_cmap(palette_name)(np.linspace(0, 1, 20))
                colors = list(base_colors) * (len(sorted_values) // 20 + 1)
                colors = colors[:len(sorted_values)]

            for i, value in enumerate(sorted_values):
                self.master_colors[field_name][value] = colors[i]

            print(f"  - {field_name}: {len(sorted_values)} unique values mapped")

    def analyze_embedding_space(self):
        """Performs t-SNE visualization on the embedding space."""
        print("\n" + "="*80)
        print("Cell2Sentence Embedding Space Analysis")
        print("="*80)

        if self.embeddings is None:
            print("Embeddings not loaded. Skipping this analysis.")
            return None

        # Extract embeddings
        embeddings, labels = self.extract_cell_embeddings()

        if len(embeddings) == 0:
            print("ERROR: No valid embeddings found.")
            return None

        # Build color mappings
        self._build_color_mappings(labels)

        print(f"\nTotal cells for analysis: {len(embeddings)}")
        print(f"Embedding shape: {embeddings.shape}")

        print("Performing t-SNE dimensionality reduction (this may take a while)...")
        tsne = TSNE(
            n_components=2,
            verbose=1,
            perplexity=min(50, len(embeddings) - 1),
            max_iter=1000,
            learning_rate='auto',
            init='pca',
            random_state=42
        )
        tsne_results = tsne.fit_transform(embeddings)

        # Create DataFrame with results
        df_data = {
            'tsne1': tsne_results[:, 0],
            'tsne2': tsne_results[:, 1],
        }

        # Add all available metadata fields
        for field in ['cell_type', 'tissue', 'batch', 'dataset', 'cell_name']:
            df_data[field] = [label.get(field, 'unknown') for label in labels]

        df = pd.DataFrame(df_data)

        # Save t-SNE results
        tsne_output = os.path.join(self.output_dir, 'tsne_results.csv')
        df.to_csv(tsne_output, index=False)
        print(f"Saved t-SNE results to: {tsne_output}")

        # Generate visualizations
        self.visualize_embedding_by_cell_type(df)

        # Visualize by other fields if available
        for field in ['tissue', 'batch', 'dataset']:
            if field in df.columns and df[field].nunique() > 1:
                self.visualize_embedding_by_field(df, field)

        return df

    def visualize_embedding_by_cell_type(self, df):
        """Visualizes embeddings colored by cell type."""
        print("Generating visualization by cell type...")

        # Filter to top cell types if specified
        if self.top_cell_types > 0:
            top_cell_types = df['cell_type'].value_counts().nlargest(self.top_cell_types).index
            filtered_df = df[df['cell_type'].isin(top_cell_types)].copy()
        else:
            filtered_df = df.copy()
            top_cell_types = df['cell_type'].unique()

        print(f"Showing {len(top_cell_types)} cell types")

        # Create figure
        fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(18, 10),
                                                 gridspec_kw={'width_ratios': [3, 1]})

        # Get color mapping
        color_map = self.master_colors.get('cell_type', {})

        # Plot each cell type
        for cell_type in top_cell_types:
            mask = filtered_df['cell_type'] == cell_type
            if mask.any():
                color = color_map.get(cell_type, 'gray')
                ax_main.scatter(filtered_df[mask]['tsne1'], filtered_df[mask]['tsne2'],
                               c=[color], label=cell_type, s=30, alpha=0.6,
                               edgecolors='white', linewidth=0.3)

        ax_main.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax_main.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax_main.set_title(f'Cell2Sentence Embedding Space by Cell Type\\n(t-SNE Visualization)\\n'
                         f'{("Top " + str(self.top_cell_types) + " cell types") if self.top_cell_types > 0 else "All cell types"}',
                         fontsize=16, fontweight='bold')
        ax_main.grid(True, linestyle='--', alpha=0.4)

        # Legend panel
        ax_legend.axis('off')
        ax_legend.text(0.05, 0.95, 'Cell Types:', fontsize=14, fontweight='bold',
                      transform=ax_legend.transAxes)

        y_pos = 0.88
        cell_type_counts = filtered_df['cell_type'].value_counts()
        for cell_type in top_cell_types[:30]:  # Limit legend entries
            if y_pos > 0.1:
                color = color_map.get(cell_type, 'gray')
                ax_legend.scatter(0.1, y_pos, c=[color], s=60, marker='o',
                                 transform=ax_legend.transAxes)
                display_name = cell_type if len(str(cell_type)) <= 25 else str(cell_type)[:22] + "..."
                count = cell_type_counts.get(cell_type, 0)
                ax_legend.text(0.2, y_pos, f"{display_name} ({count})",
                              fontsize=9, va='center', transform=ax_legend.transAxes)
                y_pos -= 0.03

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'embedding_by_cell_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Cell type visualization saved to: {output_path}")

        plt.close()

    def visualize_embedding_by_field(self, df, field_name):
        """Visualizes embeddings colored by a specific metadata field."""
        print(f"Generating visualization by {field_name}...")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Get unique values and assign colors
        unique_values = df[field_name].unique()
        color_map = self.master_colors.get(field_name, {})

        for value in unique_values:
            if value == 'unknown':
                continue
            mask = df[field_name] == value
            color = color_map.get(value, 'gray')
            ax.scatter(df[mask]['tsne1'], df[mask]['tsne2'],
                      c=[color], label=value, s=30, alpha=0.6,
                      edgecolors='white', linewidth=0.3)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.set_title(f'Cell2Sentence Embedding Space by {field_name.capitalize()}\\n(t-SNE Visualization)',
                    fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'embedding_by_{field_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{field_name.capitalize()} visualization saved to: {output_path}")

    def run_analysis(self):
        """Runs the complete analysis pipeline."""
        print("="*80)
        print("Cell2Sentence Embedding Distribution Analysis")
        print("="*80)

        if not self.load_h5ad_data():
            print("Analysis aborted due to data loading failure.")
            return

        if not self.load_model_embeddings():
            print("Analysis aborted due to model loading failure.")
            return

        self.analyze_embedding_space()

        print("\n" + "="*80)
        print("Analysis Finished!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cell2Sentence Embedding Distribution Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to h5ad file containing single-cell data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Cell2Sentence model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output visualizations')
    parser.add_argument('--n_genes', type=int, default=200,
                       help='Number of top genes to use per cell')
    parser.add_argument('--top_cell_types', type=int, default=-1,
                       help='Number of top cell types to show (-1 for all)')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='Maximum number of cells to analyze (-1 for all)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    analyzer = C2SEmbeddingDistributionAnalyzer(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_genes=args.n_genes,
        top_cell_types=args.top_cell_types,
        max_samples=args.max_samples
    )

    analyzer.run_analysis()
