"""
Visualizes Final Multi-Modal Foundation Autoencoder Results:
1. Joint Latent Space Alignment (Modality Alignment & ADR Phenotype Clustering via UMAP/t-SNE)
2. Dual-Decoder Evaluation (Gene Expression Reconstruction + Multi-Label ADR PR Curves)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, average_precision_score

# Try UMAP first for speed & global structure preservation
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

project_root = Path(__file__).resolve().parent
output_dir = project_root / "output" / "autoencoder" / "final_multimodal_autoencoder"
fig_dir = output_dir / "plots"
fig_dir.mkdir(parents=True, exist_ok=True)

latent_path = output_dir / "final_latents.npz"
metadata_path = output_dir / "final_autoencoder.json"
y_path = project_root / "Y_train_sparse.npz"
adr_vocab_path = project_root / "adr_vocabulary.txt"


def load_adr_labels(selected_cols_idx, vocab_path):
    if vocab_path.exists():
        vocab = vocab_path.read_text().splitlines()
        return [vocab[i] if i < len(vocab) else f"ADR_{i}" for i in selected_cols_idx]
    return [f"ADR_{i}" for i in selected_cols_idx]


def plot_joint_latent_space(latents, selected_rows, selected_cols, adr_names, y_sparse_path, modality_labels=None, n_samples=5000):
    """
    Visualization 1: Joint Multi-Modal Bottleneck Space
    Panel A: Colored by Modality (Verifies cross-modal alignment)
    Panel B/C: Colored by Top Adverse Events (Verifies biological phenotype clustering)
    """
    print("\n--- Generating Visualization 1: Joint Latent Space Projections ---")
    
    if len(latents) > n_samples:
        indices = np.random.choice(len(latents), size=n_samples, replace=False)
        latents_sub = latents[indices]
        rows_sub = selected_rows[indices]
        mod_sub = modality_labels[indices] if modality_labels is not None else None
    else:
        latents_sub = latents
        rows_sub = selected_rows
        mod_sub = modality_labels

    if HAS_UMAP:
        print("Projecting Latent Bottleneck to 2D via UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(latents_sub)
        algo_name = "UMAP"
    else:
        print("Projecting Latent Bottleneck to 2D via t-SNE...")
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(latents_sub)
        algo_name = "t-SNE"

    # Load Y ground truth for clinical overlays
    Y_full = sparse.load_npz(y_sparse_path).tocsr()
    Y_sub = Y_full[rows_sub][:, selected_cols].toarray()
    
    top_2_idx = np.argsort(Y_sub.sum(axis=0))[::-1][:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Modality Alignment (if multi-modal sources are passed)
    if mod_sub is not None:
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1], hue=mod_sub,
            palette="Set2", alpha=0.7, s=15, ax=axes[0]
        )
        axes[0].set_title("Modality Alignment in Bottleneck", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "Single Modality / Joint Input", ha='center', va='center')
        axes[0].set_title("Bottleneck Representation", fontweight="bold")

    axes[0].set_xlabel(f"{algo_name} 1")
    axes[0].set_ylabel(f"{algo_name} 2")

    # Panel 2 & 3: Clinical Phenotype Clusters
    for ax, idx in zip(axes[1:], top_2_idx):
        label_vec = Y_sub[:, idx]
        adr_name = adr_names[idx]
        
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1], hue=label_vec,
            palette={0: "#e0e0e0", 1: "#e63946"}, alpha=0.6, s=15, ax=ax, legend=False
        )
        ax.set_title(f"Clinical Phenotype: {adr_name}", fontweight="bold")
        ax.set_xlabel(f"{algo_name} 1")
        ax.set_ylabel(f"{algo_name} 2")

    plt.suptitle(f"Final Foundation Model Bottleneck ({algo_name} Space)", fontsize=14, fontweight="bold", y=1.02)
    save_fig = fig_dir / "viz1_joint_latent_space.png"
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_fig}")


def plot_dual_reconstruction_performance(metadata_file):
    """
    Visualization 2: Dual Decoder Performance Metrics
    Panel A: Gene Expression Reconstruction (Transcriptomic MSE / R2)
    Panel B: Clinical ADR Precision-Recall Buckets
    """
    print("\n--- Generating Visualization 2: Dual-Decoder Performance ---")
    
    with open(metadata_file, "r") as f:
        meta = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Transcriptomic Expression Loss Curve / Metrics
    history = meta.get("history", {})
    if "train_loss" in history and "val_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1d3557", lw=2)
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#e63946", lw=2, linestyle="--")
        ax1.set_title("Multi-Modal Loss Convergence", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

    # Panel B: Clinical ADR PR Scores by Frequency Bucket
    buckets = meta.get("final_evaluation", {}).get("frequency_buckets", [])
    if buckets:
        df_buckets = pd.DataFrame(buckets)
        df_melted = df_buckets.melt(
            id_vars=["bucket"],
            value_vars=["precision", "recall", "f1", "auc_pr"],
            var_name="Metric",
            value_name="Score"
        )
        df_melted["Metric"] = df_melted["Metric"].str.upper()

        sns.barplot(
            data=df_melted, x="bucket", y="Score", hue="Metric",
            palette="Spectral", ax=ax2
        )
        ax2.set_title("Clinical ADR Reconstruction by Frequency", fontweight="bold")
        ax2.set_xlabel("ADR Rarity Bucket")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    save_fig = fig_dir / "viz2_dual_reconstruction_metrics.png"
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_fig}")


def main():
    if not latent_path.exists() or not metadata_path.exists():
        print(f"Missing final autoencoder run files in {output_dir}")
        return

    npz_data = np.load(latent_path)
    latents = npz_data["latent_embeddings"]
    selected_rows = npz_data["selected_row_indices"]
    selected_cols = npz_data["selected_adr_columns"]
    modality_labels = npz_data.get("modality_labels", None)

    adr_names = load_adr_labels(selected_cols, adr_vocab_path)

    plot_joint_latent_space(latents, selected_rows, selected_cols, adr_names, y_path, modality_labels)
    plot_dual_reconstruction_performance(metadata_path)

    print(f"\nFinal Autoencoder Visualizations complete! Figures saved in:\n  {fig_dir}")


if __name__ == "__main__":
    main()