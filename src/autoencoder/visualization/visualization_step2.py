"""
Visualizes Step 2 Chemical Baseline Results:
1. Training Loss Convergence & Validation Metrics across Epochs
2. UMAP/t-SNE Latent Space Projections (colored by ADR ground truth)
3. Precision-Recall Curves & Frequency Bucket Performance
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse
from sklearn.manifold import TSNE

# Try importing UMAP; fallback to t-SNE if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Set plotting style matching your image
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Setup Paths
project_root = Path(__file__).resolve().parent
output_dir = project_root / "output" / "autoencoder" / "drug_adr_encoder"
fig_dir = output_dir / "plots"
fig_dir.mkdir(parents=True, exist_ok=True)

latent_path = "/Users/duncanpark/10-faers-foundation-model/output/autoencoder/drug_adr_encoder/drug_adr_latents.npz"
metadata_path = "/Users/duncanpark/10-faers-foundation-model/output/autoencoder/drug_adr_encoder/drug_adr_encoder.json"
adr_vocab_path = "/Users/duncanpark/10-faers-foundation-model/adr_vocabulary.txt"
y_path = project_root / "Y_train_sparse.npz"


def load_adr_labels(selected_cols_idx, vocab_path):
    """Maps selected column indices back to text ADR names."""
    path_obj = Path(vocab_path)
    if path_obj.exists():
        vocab = path_obj.read_text().splitlines()
        return [vocab[i] if i < len(vocab) else f"ADR_{i}" for i in selected_cols_idx]
    return [f"ADR_{i}" for i in selected_cols_idx]


def plot_training_history(meta):
    """
    Visualization 0: Loss Optimization History & Validation Metrics Tracking
    (Matches the 2-panel epoch tracking format)
    """
    print("\n--- Generating Visualization 0: Training History & Metrics Tracking ---")
    
    history = meta.get("history", {})
    if not history or "train_loss" not in history:
        print("No training history found in metadata. Skipping loss visualization.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = np.argmin(history["val_loss"]) + 1 if "val_loss" in history else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Loss Optimization History
    ax1.plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], marker="s", linestyle="--", label="Val Loss", color="#ff7f0e", lw=2)
    if best_epoch:
        ax1.axvline(best_epoch, color="red", linestyle=":", label=f"Best Checkpoint (Epoch {best_epoch})")

    ax1.set_title("BCE Loss Optimization History", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs)
    ax1.legend(loc="upper right")

    # Panel B: Validation Metrics Tracking
    if "val_precision" in history and "val_recall" in history and "val_f1" in history:
        ax2.plot(epochs, history["val_precision"], marker="^", label="Precision", color="#2ca02c", lw=2)
        ax2.plot(epochs, history["val_recall"], marker="v", label="Recall", color="#d62728", lw=2)
        ax2.plot(epochs, history["val_f1"], marker="D", label="F1-Score", color="#9467bd", lw=2)
        
        ax2.set_title("Validation Metrics Tracking", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score Matrix")
        ax2.set_xticks(epochs)
        ax2.legend(loc="upper right")

    plt.tight_layout()
    save_fig = fig_dir / "viz0_training_history.png"
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_fig}")


def plot_latent_space(latents, selected_rows, selected_cols, adr_names, y_sparse_path, n_samples=5000):
    """Visualization 1: 2D Projection (UMAP or t-SNE) of the Latent Space."""
    print("\n--- Generating Visualization 1: Latent Space Projection ---")
    
    if len(latents) > n_samples:
        indices = np.random.choice(len(latents), size=n_samples, replace=False)
        latents_sub = latents[indices]
        rows_sub = selected_rows[indices]
    else:
        latents_sub = latents
        rows_sub = selected_rows

    if HAS_UMAP:
        print("Projecting 512D -> 2D using UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(latents_sub)
        algo_name = "UMAP"
    else:
        print("UMAP not found. Projecting 512D -> 2D using t-SNE...")
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(latents_sub)
        algo_name = "t-SNE"

    if not Path(y_sparse_path).exists():
        print(f"Warning: {y_sparse_path} not found. Skipping latent point color overlays.")
        return

    Y_full = sparse.load_npz(y_sparse_path).tocsr()
    Y_sub = Y_full[rows_sub][:, selected_cols].toarray()

    adr_counts = Y_sub.sum(axis=0)
    top_3_idx = np.argsort(adr_counts)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, idx in zip(axes, top_3_idx):
        label_vec = Y_sub[:, idx]
        adr_name = adr_names[idx]
        
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=label_vec,
            palette={0: "#e0e0e0", 1: "#e63946"},
            alpha=0.6,
            s=15,
            ax=ax,
            legend=False
        )
        ax.set_title(f"Latent Clusters: {adr_name}", fontweight="bold")
        ax.set_xlabel(f"{algo_name} 1")
        ax.set_ylabel(f"{algo_name} 2")

    plt.suptitle(f"Chemical Encoder 512-Dim Bottleneck ({algo_name} Projection)", fontsize=14, fontweight="bold", y=1.02)
    save_fig = fig_dir / "viz1_latent_space_projection.png"
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_fig}")


def plot_pr_curves_and_buckets(meta):
    """Visualization 2: PR Curves & Performance Across Rare vs Common ADR Buckets."""
    print("\n--- Generating Visualization 2: PR Curves & Frequency Buckets ---")

    buckets = meta.get("final_evaluation", {}).get("frequency_buckets", [])
    if not buckets:
        print("No frequency bucket metadata found. Skipping Viz 2.")
        return

    df_buckets = pd.DataFrame(buckets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def bucket_metric_error(score: float, support: int) -> float:
        """Approximate a standard error for display purposes."""
        support = max(int(support), 1)
        score = float(np.clip(score, 0.0, 1.0))
        return float(np.sqrt(max(score * (1.0 - score), 1e-6) / support))

    # Panel A: Metrics by Frequency Bucket
    df_melted = df_buckets.melt(
        id_vars=["bucket", "n_columns"],
        value_vars=["precision", "recall", "f1", "auc_pr"],
        var_name="Metric",
        value_name="Score"
    )
    df_melted["Metric"] = df_melted["Metric"].str.upper()

    sns.barplot(
        data=df_melted,
        x="bucket",
        y="Score",
        hue="Metric",
        palette="Blues_d",
        ax=ax1
    )
    ax1.set_title("Performance by ADR Frequency Bucket", fontweight="bold")
    ax1.set_xlabel("ADR Frequency Bucket")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)

    metric_order = ["PRECISION", "RECALL", "F1", "AUC_PR"]
    for container, metric in zip(ax1.containers, metric_order):
        for idx, bar in enumerate(container):
            if idx >= len(df_buckets):
                continue
            row = df_buckets.iloc[idx]
            score = float(bar.get_height())
            if score <= 0:
                continue
            err = bucket_metric_error(score, row["n_columns"])
            ax1.errorbar(
                bar.get_x() + bar.get_width() / 2,
                score,
                yerr=err,
                fmt="none",
                ecolor="#1f1f1f",
                elinewidth=1,
                capsize=3,
                capthick=1,
                zorder=3,
            )

    # Panel B: Threshold Sweep Curves
    sweep = meta.get("final_evaluation", {}).get("threshold_sweep", [])
    if sweep:
        df_sweep = pd.DataFrame(sweep)
        ax2.plot(df_sweep["threshold"], df_sweep["precision"], label="Precision", color="#1d3557", lw=2)
        ax2.plot(df_sweep["threshold"], df_sweep["recall"], label="Recall", color="#e63946", lw=2)
        ax2.plot(df_sweep["threshold"], df_sweep["f1"], label="F1 Score", color="#2a9d8f", lw=2, linestyle="--")

        best_t = meta["final_evaluation"]["best_threshold"]["threshold"]
        ax2.axvline(best_t, color="black", linestyle=":", label=f"Best Threshold ({best_t:.2f})")

        ax2.set_title("Precision-Recall-F1 Trade-off Across Decision Thresholds", fontweight="bold")
        ax2.set_xlabel("Decision Threshold")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower left")

    plt.tight_layout()
    save_fig = fig_dir / "viz2_pr_curves_and_buckets.png"
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_fig}")


def main():
    if not Path(latent_path).exists():
        print(f"Error: Latent file not found at {latent_path}")
        return

    # Load artifacts
    npz_data = np.load(latent_path)
    latents = npz_data["latent_embeddings"]
    selected_rows = npz_data["selected_row_indices"]
    selected_cols = npz_data["selected_adr_columns"]

    adr_names = load_adr_labels(selected_cols, adr_vocab_path)

    meta_path = Path(metadata_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # Generate visualizers
    plot_training_history(meta)
    plot_latent_space(latents, selected_rows, selected_cols, adr_names, y_path)
    plot_pr_curves_and_buckets(meta)

    print("\nVisualizations completed successfully! Check the output directory:")
    print(f"  {fig_dir}")


if __name__ == "__main__":
    main()
