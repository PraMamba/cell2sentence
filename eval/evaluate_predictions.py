"""
Cell Type Prediction Evaluation Script

Evaluates cell type predictions using user-defined mapping dictionaries.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================
# Cell Type Mapping Dictionaries
# ============================================

# A013 Dataset: Immune cells
celltype_mapping_A013 = [
    {"raw_name": "alpha-beta memory T cell", "standardized_name": "Alpha-beta memory T cell", "broad_category": "T cell"},
    {"raw_name": "alpha-beta T cell", "standardized_name": "Alpha-beta T cell", "broad_category": "T cell"},
    {"raw_name": "CD14-low", "standardized_name": "CD14-low monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD14-positive monocyte", "standardized_name": "CD14+ monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD16-negative", "standardized_name": "CD16-negative monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD16-negative classical monocyte", "standardized_name": "CD16-negative classical monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD56-dim natural killer cell", "standardized_name": "CD56^dim NK cell", "broad_category": "NK cell"},
    {"raw_name": "central memory CD4-positive", "standardized_name": "Central memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "conventional dendritic cell", "standardized_name": "Conventional dendritic cell", "broad_category": "Dendritic cell"},
    {"raw_name": "effector memory CD4-positive", "standardized_name": "Effector memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "effector memory CD8-positive", "standardized_name": "Effector memory CD8+ T cell", "broad_category": "T cell"},
    {"raw_name": "gamma-delta T cell", "standardized_name": "Gamma-delta T cell", "broad_category": "T cell"},
    {"raw_name": "memory B cell", "standardized_name": "Memory B cell", "broad_category": "B cell"},
    {"raw_name": "mucosal invariant T cell", "standardized_name": "MAIT cell", "broad_category": "T cell"},
    {"raw_name": "naive B cell", "standardized_name": "Naive B cell", "broad_category": "B cell"},
    {"raw_name": "natural killer cell", "standardized_name": "NK cell", "broad_category": "NK cell"},
    {"raw_name": "plasmablast", "standardized_name": "Plasmablast", "broad_category": "B cell"},
    {"raw_name": "plasmacytoid dendritic cell", "standardized_name": "Plasmacytoid dendritic cell", "broad_category": "Dendritic cell"},
    {"raw_name": "transitional stage B cell", "standardized_name": "Transitional B cell", "broad_category": "B cell"},
]

# D099 Dataset: Epithelial cells (Predicted column)
celltype_mapping_D099 = [
    {"raw_name": "basal cell", "standardized_name": "Basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epidermis", "standardized_name": "Epidermal basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epithelium of bronchus", "standardized_name": "Bronchial basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epithelium of trachea", "standardized_name": "Tracheal basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of prostate epithelium", "standardized_name": "Prostate basal cell", "broad_category": "Epithelial"},
    {"raw_name": "blood vessel endothelial cell", "standardized_name": "Vascular endothelial cell", "broad_category": "Endothelial"},
    {"raw_name": "brush cell", "standardized_name": "Brush cell", "broad_category": "Epithelial"},
    {"raw_name": "cell in vitro", "standardized_name": "Cultured cell", "broad_category": "Other"},
    {"raw_name": "ciliated cell", "standardized_name": "Ciliated epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "ciliated cell of the bronchus", "standardized_name": "Bronchial ciliated cell", "broad_category": "Epithelial"},
    {"raw_name": "club cell", "standardized_name": "Club cell", "broad_category": "Epithelial"},
    {"raw_name": "corneal epithelial cell", "standardized_name": "Corneal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "cultured cell", "standardized_name": "Cultured cell", "broad_category": "Other"},
    {"raw_name": "duct epithelial cell", "standardized_name": "Ductal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "endothelial cell of lymphatic vessel", "standardized_name": "Lymphatic endothelial cell", "broad_category": "Endothelial"},
    {"raw_name": "endothelial cell of umbilical vein", "standardized_name": "Umbilical vein endothelial cell (HUVEC)", "broad_category": "Endothelial"},
    {"raw_name": "epithelial cell", "standardized_name": "Epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of esophagus", "standardized_name": "Esophageal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of lower respiratory tract", "standardized_name": "Lower respiratory epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of lung", "standardized_name": "Pulmonary epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of nephron", "standardized_name": "Nephron epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of proximal tubule", "standardized_name": "Proximal tubule epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of the bronchus", "standardized_name": "Bronchial epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of urethra", "standardized_name": "Urethral epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "foveolar cell of stomach", "standardized_name": "Gastric foveolar cell", "broad_category": "Epithelial"},
    {"raw_name": "goblet cell", "standardized_name": "Goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "granulocyte", "standardized_name": "Granulocyte", "broad_category": "Immune"},
    {"raw_name": "kidney collecting duct principal cell", "standardized_name": "Collecting duct principal cell", "broad_category": "Epithelial"},
    {"raw_name": "kidney loop of Henle thin ascending limb epithelial cell", "standardized_name": "Loop of Henle thin ascending limb cell", "broad_category": "Epithelial"},
    {"raw_name": "kidney loop of Henle thin descending limb epithelial cell", "standardized_name": "Loop of Henle thin descending limb cell", "broad_category": "Epithelial"},
    {"raw_name": "lung ciliated cell", "standardized_name": "Pulmonary ciliated cell", "broad_category": "Epithelial"},
    {"raw_name": "lung goblet cell", "standardized_name": "Pulmonary goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "lung secretory cell", "standardized_name": "Pulmonary secretory cell", "broad_category": "Epithelial"},
    {"raw_name": "malignant cell", "standardized_name": "Malignant cell", "broad_category": "Other"},
    {"raw_name": "melanocyte", "standardized_name": "Melanocyte", "broad_category": "Pigment"},
    {"raw_name": "nasal mucosa goblet cell", "standardized_name": "Nasal goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "native cell", "standardized_name": "Native cell", "broad_category": "Other"},
    {"raw_name": "neutrophil", "standardized_name": "Neutrophil", "broad_category": "Immune"},
    {"raw_name": "respiratory basal cell", "standardized_name": "Respiratory basal cell", "broad_category": "Epithelial"},
    {"raw_name": "secretory cell", "standardized_name": "Secretory epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "squamous epithelial cell", "standardized_name": "Squamous epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "stratified epithelial cell", "standardized_name": "Stratified epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "tracheal goblet cell", "standardized_name": "Tracheal goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "type II pneumocyte", "standardized_name": "Type II pneumocyte (AT2 cell)", "broad_category": "Epithelial"},
]

# D099 Dataset: State labels (GroundTruth column)
state_mapping_D099 = [
    {"raw_name": "Basal", "standardized_name": "Basal"},
    {"raw_name": "Ciliated", "standardized_name": "Ciliated"},
    {"raw_name": "Differentiating.Basal", "standardized_name": "Differentiating basal"},
    {"raw_name": "Proliferating.Basal", "standardized_name": "Proliferating basal"},
    {"raw_name": "Secretory", "standardized_name": "Secretory"},
    {"raw_name": "Suprabasal", "standardized_name": "Suprabasal"},
    {"raw_name": "Transitioning.Basal", "standardized_name": "Transitioning basal"},
]


def build_normalization_dict(mapping_list):
    """Build dictionary for normalization."""
    norm_dict = {}
    for item in mapping_list:
        raw = item["raw_name"].lower().strip().replace('"', '').replace("'", "")
        norm_dict[raw] = item["standardized_name"]
    return norm_dict


def normalize_cell_type(cell_type, norm_dict):
    """Normalize cell type using mapping dictionary."""
    if pd.isna(cell_type) or cell_type == "":
        return "Unknown"
    
    cell_type_clean = str(cell_type).strip().lower().replace('"', '').replace("'", "")
    
    if cell_type_clean in norm_dict:
        return norm_dict[cell_type_clean]
    
    for key, value in norm_dict.items():
        if key in cell_type_clean:
            return value
    
    return str(cell_type).strip()


def normalize_state_label(state_label):
    """Normalize D099 state labels."""
    if pd.isna(state_label) or state_label == "":
        return "Unknown"
    
    state_clean = str(state_label).strip()
    
    for item in state_mapping_D099:
        if item["raw_name"] == state_clean:
            return item["standardized_name"]
    
    return state_clean


def load_and_normalize_A013(file_path):
    """Load and normalize A013 dataset."""
    df = pd.read_csv(file_path)
    norm_dict = build_normalization_dict(celltype_mapping_A013)
    
    df["Predicted_Normalized"] = df["Predicted"].apply(lambda x: normalize_cell_type(x, norm_dict))
    df["GroundTruth_Normalized"] = df["GroundTruth"].apply(lambda x: normalize_cell_type(x, norm_dict))
    
    return df


def load_and_normalize_D099(file_path):
    """Load and normalize D099 dataset."""
    df = pd.read_csv(file_path)
    pred_norm_dict = build_normalization_dict(celltype_mapping_D099)
    
    df["Predicted_Normalized"] = df["Predicted"].apply(lambda x: normalize_cell_type(x, pred_norm_dict))
    df["GroundTruth_Normalized"] = df["GroundTruth"].apply(normalize_state_label)
    
    return df


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix", normalize=False):
    """Plot and save confusion matrix with visible heatmap color."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
    else:
        fmt = "d"
    
    # Set color range properly (avoid all-white)
    vmax = np.max(cm)
    vmin = 0 if not normalize else 0.0
    
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.7), max(10, len(labels) * 0.6)))
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Reds",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True, square=True, linewidths=0.5,
        vmin=vmin, vmax=vmax
    )
    
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title(title, fontsize=18, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# def plot_confusion_matrix(y_true, y_pred, output_path, top_n=25, title="Top-N Confusion Matrix"):
#     labels = sorted(list(set(y_true) | set(y_pred)))
#     cm = confusion_matrix(y_true, y_pred, labels=labels)

#     # 选出样本最多的前 N 个类别
#     sums = cm.sum(axis=1)
#     top_idx = np.argsort(sums)[-top_n:]
#     cm_top = cm[top_idx][:, top_idx]
#     labels_top = [labels[i] for i in top_idx]

#     fig, ax = plt.subplots(figsize=(max(12, len(labels_top) * 0.7), max(10, len(labels_top) * 0.6)))
#     sns.heatmap(cm_top, annot=True, fmt="d", cmap="Reds",
#                 xticklabels=labels_top, yticklabels=labels_top,
#                 ax=ax, cbar=True, square=True, linewidths=0.5)
#     ax.set_title(title, fontsize=14, fontweight="bold")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()


def save_classification_report(y_true, y_pred, output_path):
    """Save classification report."""
    report = classification_report(y_true, y_pred, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)


def save_mapping_analysis(df, output_path, dataset_name):
    """Save cell type mapping analysis."""
    pred_mapping = df[["Predicted", "Predicted_Normalized"]].drop_duplicates().sort_values("Predicted_Normalized")
    gt_mapping = df[["GroundTruth", "GroundTruth_Normalized"]].drop_duplicates().sort_values("GroundTruth_Normalized")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Cell Type Mapping Analysis - {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Predicted → Normalized:\n")
        f.write("-" * 80 + "\n")
        for _, row in pred_mapping.iterrows():
            f.write(f"{row['Predicted']:60s} → {row['Predicted_Normalized']}\n")
        
        f.write("\n\nGroundTruth → Normalized:\n")
        f.write("-" * 80 + "\n")
        for _, row in gt_mapping.iterrows():
            f.write(f"{row['GroundTruth']:60s} → {row['GroundTruth_Normalized']}\n")
        
        f.write("\n\nStatistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Unique predicted types (normalized): {df['Predicted_Normalized'].nunique()}\n")
        f.write(f"Unique ground truth types (normalized): {df['GroundTruth_Normalized'].nunique()}\n")


def evaluate_dataset(file_path, output_dir, dataset_name, load_func):
    """Evaluate dataset."""
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'=' * 80}\n")
    
    df = load_func(file_path)
    
    # Save normalized data
    df.to_csv(os.path.join(output_dir, f"{dataset_name}_normalized.csv"), index=False)
    
    # Save mapping
    save_mapping_analysis(df, os.path.join(output_dir, f"{dataset_name}_mapping.txt"), dataset_name)
    
    # Calculate metrics
    y_true = df["GroundTruth_Normalized"]
    y_pred = df["Predicted_Normalized"]
    metrics = calculate_metrics(y_true, y_pred)
    
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, f"{dataset_name}_metrics.txt"), "w") as f:
        f.write(f"Metrics - {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"), 
                         title=f"Confusion Matrix - {dataset_name}")
    
    # Classification report
    save_classification_report(y_true, y_pred, os.path.join(output_dir, f"{dataset_name}_classification_report.txt"))
    
    return df, metrics


def main():
    """Main evaluation function."""
    base_dir = "/home/scbjtfy/cell2sentence"
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CELL TYPE PREDICTION EVALUATION")
    print("=" * 80)
    
    # Evaluate A013
    a013_file = os.path.join(base_dir, "tutorials", "c2s_A013_predict.csv")
    a013_df, a013_metrics = evaluate_dataset(a013_file, eval_dir, "A013", load_and_normalize_A013)
    
    # Evaluate D099
    d099_file = os.path.join(base_dir, "tutorials", "c2s_D099_predict.csv")
    d099_df, d099_metrics = evaluate_dataset(d099_file, eval_dir, "D099", load_and_normalize_D099)
    
    print(f"\n{'=' * 80}")
    print("Evaluation completed!")
    print(f"{'=' * 80}\n")
    print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()






