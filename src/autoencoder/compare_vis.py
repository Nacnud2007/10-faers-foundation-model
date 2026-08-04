"""
Compare the multimodal ADR predictor (autoencoder.py -> MultimodalADRPredictor,
chem + transcriptomic) against the chemical-only baseline
(drug_adr_encoder.py -> DrugToAdverseEventAutoencoder, chem only).

Includes bootstrap evaluation to generate standard deviations and plot error bars.

Example:
  python src/autoencoder/compare_checkpoints.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from autoencoder import (
    BatchedBimodalLoader,
    BimodalDataset,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    MultimodalADRPredictor,
    get_autocast_context,
    load_sparse_npz_fast,
)
from drug_adr_encoder import (
    DrugToAdverseEventAutoencoder,
    SparseDrugEventDataset,
    split_indices,
    sweep_thresholds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIMODAL_CHECKPOINT = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "best_val.pt"
DEFAULT_CHEM_DIR = PROJECT_ROOT / "output" / "autoencoder" / "drug_adr_encoder"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "analysis"


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_multimodal(
    checkpoint_path: Path,
    x_path: Path,
    y_path: Path,
    trans_profiles_path: Path,
    trans_patient_indices_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Reconstruct MultimodalADRPredictor's own val split and run inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    validation_fraction = float(ckpt_args.get("validation_fraction", 0.1))
    split_seed = int(ckpt_args.get("split_seed", 0))
    trans_scale = float(ckpt_args.get("trans_scale", 1.0))
    dropout = float(ckpt_args.get("dropout", 0.0))
    max_rows = ckpt_args.get("max_rows", None)
    selected_adr_columns = checkpoint["selected_adr_columns"]

    chem_matrix = load_sparse_npz_fast(x_path)
    clin_matrix = load_sparse_npz_fast(y_path)
    trans_profiles = np.load(trans_profiles_path, mmap_mode="r")
    trans_patient_indices = np.load(trans_patient_indices_path, mmap_mode="r")

    # Top-k ADR column selection -- reuse the exact columns saved in the checkpoint.
    clin_matrix = clin_matrix[:, selected_adr_columns].tocsr()

    # Qualifying-row filter: at least one selected ADR present.
    qualifying_rows = np.flatnonzero(np.asarray(clin_matrix.sum(axis=1)).ravel() > 0)

    # Max-rows subsample, reproduced with the same seeded RNG call as main().
    if max_rows is not None and max_rows < len(qualifying_rows):
        rng = np.random.default_rng(split_seed)
        selected_row_indices = rng.choice(qualifying_rows, size=max_rows, replace=False)
        selected_row_indices.sort()
    else:
        selected_row_indices = qualifying_rows

    chem_matrix = chem_matrix[selected_row_indices]
    clin_matrix = clin_matrix[selected_row_indices]
    trans_patient_indices = trans_patient_indices[selected_row_indices]

    non_zero_per_row = np.diff(chem_matrix.indptr)
    valid_indices = np.where(non_zero_per_row > 0)[0]
    _, val_rel = split_indices(len(valid_indices), validation_fraction, split_seed)
    val_indices = valid_indices[val_rel]

    dataset = BimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)
    val_loader = BatchedBimodalLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        indices=val_indices,
        trans_scale=trans_scale,
        chemical_as_sparse=True,
    )

    model = MultimodalADRPredictor(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_chunks, y_prob_chunks = [], []
    with torch.inference_mode():
        for chem_batch, trans_batch, trans_mask_batch, clin_batch in val_loader:
            if isinstance(chem_batch, torch.Tensor):
                chem_batch = chem_batch.to(device)
            trans_batch = trans_batch.to(device)
            trans_mask_batch = trans_mask_batch.to(device)
            clin_batch = clin_batch.to(device)
            with get_autocast_context(device):
                logits, _ = model(chem_batch, trans_batch, trans_mask_batch)
            y_true_chunks.append(clin_batch.float().cpu().numpy())
            y_prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

    return {
        "name": "Multimodal (chem+trans)",
        "y_true": np.vstack(y_true_chunks),
        "y_prob": np.vstack(y_prob_chunks),
        "selected_columns": np.asarray(selected_adr_columns),
        "n_val_rows": len(val_indices),
    }


def evaluate_chem_only(
    chem_dir: Path,
    x_path: Path,
    y_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Reconstruct DrugToAdverseEventAutoencoder's own val split and run inference."""
    checkpoint_path = chem_dir / "drug_adr_encoder.pt"
    metadata_path = chem_dir / "drug_adr_encoder.json"
    selected_columns_path = chem_dir / "selected_adr_columns.npy"
    latents_path = chem_dir / "drug_adr_latents.npz"

    for path in (checkpoint_path, metadata_path, selected_columns_path, latents_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing expected drug_adr_encoder.py artifact: {path}. "
                "Point --chem-dir at the output directory from that training run."
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    metadata = json.loads(metadata_path.read_text())
    validation_fraction = float(metadata["validation_fraction"])
    seed = int(metadata["seed"])

    selected_columns = np.load(selected_columns_path)
    latents = np.load(latents_path)
    selected_row_indices = latents["selected_row_indices"]

    chem_matrix = load_sparse_npz_fast(x_path)
    adverse_event_matrix = load_sparse_npz_fast(y_path)

    chem_subset = chem_matrix[selected_row_indices]
    adr_subset = adverse_event_matrix[selected_row_indices][:, selected_columns].tocsr()

    _, val_indices = split_indices(chem_subset.shape[0], validation_fraction, seed)

    dataset = SparseDrugEventDataset(chem_subset, adr_subset)
    val_subset = Subset(dataset, val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = DrugToAdverseEventAutoencoder(
        output_dim=model_config["output_dim"],
        slot_embed_dim=model_config["slot_embed_dim"],
        chemical_hidden_dim=model_config["chemical_hidden_dim"],
        latent_dim=model_config["latent_dim"],
        decoder_hidden_dim=model_config["decoder_hidden_dim"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_chunks, y_prob_chunks = [], []
    with torch.no_grad():
        for chemical_batch, adverse_event_batch in val_loader:
            chemical_batch = chemical_batch.to(device)
            logits, _ = model(chemical_batch)
            y_true_chunks.append(adverse_event_batch.numpy())
            y_prob_chunks.append(torch.sigmoid(logits).cpu().numpy())

    return {
        "name": "Chemical-only",
        "y_true": np.vstack(y_true_chunks),
        "y_prob": np.vstack(y_prob_chunks),
        "selected_columns": np.asarray(selected_columns),
        "n_val_rows": len(val_indices),
    }


def restrict_to_shared_columns(result: dict, shared_columns: np.ndarray) -> dict:
    """Slice a result's y_true/y_prob down to just the shared ADR columns,
    in shared_columns order, using its own selected_columns as the index map."""
    col_to_pos = {col: pos for pos, col in enumerate(result["selected_columns"])}
    positions = np.array([col_to_pos[c] for c in shared_columns], dtype=np.int64)
    return {
        "name": result["name"],
        "y_true": result["y_true"][:, positions],
        "y_prob": result["y_prob"][:, positions],
        "n_val_rows": result["n_val_rows"],
    }


def summarize_with_bootstrap(result: dict, n_bootstraps: int = 100, seed: int = 42) -> dict:
    """Computes mean and std across bootstrap samples of the validation set."""
    rng = np.random.default_rng(seed)
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    n_samples = y_true.shape[0]

    bootstrapped_metrics = {m: [] for m in ["precision", "recall", "f1", "ap", "roc_auc"]}

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_boot = y_true[boot_idx].reshape(-1).astype(np.int32)
        yp_boot = y_prob[boot_idx].reshape(-1)

        if len(np.unique(yt_boot)) < 2:
            continue

        _, best = sweep_thresholds(yt_boot, yp_boot)
        bootstrapped_metrics["precision"].append(best["precision"])
        bootstrapped_metrics["recall"].append(best["recall"])
        bootstrapped_metrics["f1"].append(best["f1"])
        bootstrapped_metrics["ap"].append(average_precision_score(yt_boot, yp_boot))
        try:
            bootstrapped_metrics["roc_auc"].append(roc_auc_score(yt_boot, yp_boot))
        except ValueError:
            pass

    summary = {"name": result["name"], "n_val_rows": result["n_val_rows"]}
    for metric, values in bootstrapped_metrics.items():
        summary[metric] = float(np.mean(values)) if values else float("nan")
        summary[f"{metric}_std"] = float(np.std(values)) if values else float("nan")

    return summary


def per_column_ap(result: dict, shared_columns: np.ndarray) -> pd.Series:
    aps = []
    for col_idx in range(result["y_true"].shape[1]):
        col_true = result["y_true"][:, col_idx]
        if col_true.sum() > 0:
            aps.append(average_precision_score(col_true, result["y_prob"][:, col_idx]))
        else:
            aps.append(np.nan)
    return pd.Series(aps, index=shared_columns, name=result["name"])


def make_figure(summary_df: pd.DataFrame, ap_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    metrics = ["precision", "recall", "f1", "ap", "roc_auc"]

    # Reshape means and stds for seaborn
    melted_means = summary_df.melt(id_vars=["name"], value_vars=metrics, var_name="Metric", value_name="Score")
    melted_stds = summary_df.melt(id_vars=["name"], value_vars=[f"{m}_std" for m in metrics], var_name="Metric", value_name="Std")
    melted_stds["Metric"] = melted_stds["Metric"].str.replace("_std", "")

    merged = pd.merge(melted_means, melted_stds, on=["name", "Metric"])

    # Plot baseline bars
    ax = axes[0]
    palette = sns.color_palette("Set2")
    sns.barplot(data=merged, x="Metric", y="Score", hue="name", ax=ax, palette=palette)

    # Compute coordinate offsets to overlay error bars precisely
    n_metrics = len(metrics)
    models = merged["name"].unique()
    n_hues = len(models)
    bar_width = 0.8 / n_hues

    for hue_idx, model_name in enumerate(models):
        model_data = merged[merged["name"] == model_name]
        x_coords = np.arange(n_metrics) + (hue_idx - n_hues / 2 + 0.5) * bar_width
        
        ax.errorbar(
            x=x_coords,
            y=model_data["Score"],
            yerr=model_data["Std"],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )

    ax.set_title("Overall Val Metrics on Shared ADR Columns", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(title=None)

    # Per-ADR AUC-PR Scatter Plot
    valid = ap_df.dropna()
    axes[1].scatter(valid.iloc[:, 0], valid.iloc[:, 1], alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [0.0, 1.0]
    axes[1].plot(lims, lims, color="gray", linestyle="--", lw=1)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel(f"AUC-PR: {ap_df.columns[0]}")
    axes[1].set_ylabel(f"AUC-PR: {ap_df.columns[1]}")
    axes[1].set_title("Per-ADR AUC-PR (shared columns)", fontweight="bold")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multimodal vs chemical-only ADR checkpoints.")
    parser.add_argument("--multimodal-checkpoint", type=Path, default=DEFAULT_MULTIMODAL_CHECKPOINT)
    parser.add_argument("--chem-dir", type=Path, default=DEFAULT_CHEM_DIR,
                         help="Output dir from drug_adr_encoder.py training (holds drug_adr_encoder.pt/.json etc).")
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstraps", type=int, default=100, help="Number of bootstrap samples for error bar calculation.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"\nEvaluating multimodal checkpoint: {args.multimodal_checkpoint}")
    multimodal_result = evaluate_multimodal(
        args.multimodal_checkpoint, args.x_path, args.y_path,
        args.trans_profiles, args.trans_patient_indices,
        batch_size=args.batch_size, device=device,
    )
    print(f"  Val rows: {multimodal_result['n_val_rows']:,} | ADR columns: {len(multimodal_result['selected_columns']):,}")

    print(f"\nEvaluating chemical-only checkpoint: {args.chem_dir / 'drug_adr_encoder.pt'}")
    chem_result = evaluate_chem_only(
        args.chem_dir, args.x_path, args.y_path,
        batch_size=args.batch_size, device=device,
    )
    print(f"  Val rows: {chem_result['n_val_rows']:,} | ADR columns: {len(chem_result['selected_columns']):,}")

    shared_columns = np.intersect1d(multimodal_result["selected_columns"], chem_result["selected_columns"])
    print(f"\nShared ADR columns between the two models: {len(shared_columns):,}")
    if len(shared_columns) == 0:
        raise SystemExit(
            "No overlapping ADR columns between the two checkpoints -- likely because "
            "drug_adr_encoder.py used a custom ADR list (--modified-top-100-adrs) instead "
            "of top-k-by-frequency. Nothing to compare column-for-column."
        )

    multimodal_shared = restrict_to_shared_columns(multimodal_result, shared_columns)
    chem_shared = restrict_to_shared_columns(chem_result, shared_columns)

    print(f"\nRunning bootstrap sampling (N={args.n_bootstraps}) to compute standard deviations...")
    summary_rows = [
        summarize_with_bootstrap(multimodal_shared, n_bootstraps=args.n_bootstraps),
        summarize_with_bootstrap(chem_shared, n_bootstraps=args.n_bootstraps),
    ]
    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Comparison on shared ADR columns (Bootstrapped Means & Stds) ===")
    print(summary_df.to_string(index=False))

    ap_multimodal = per_column_ap(multimodal_shared, shared_columns)
    ap_chem = per_column_ap(chem_shared, shared_columns)
    ap_df = pd.concat([ap_multimodal, ap_chem], axis=1)

    n_multimodal_better = int((ap_df.iloc[:, 0] > ap_df.iloc[:, 1]).sum())
    n_chem_better = int((ap_df.iloc[:, 1] > ap_df.iloc[:, 0]).sum())
    print(
        f"\nPer-ADR AUC-PR: multimodal higher on {n_multimodal_better}/{len(ap_df.dropna())}, "
        f"chemical-only higher on {n_chem_better}/{len(ap_df.dropna())} (of columns with positives in both)."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "model_comparison.png"
    make_figure(summary_df, ap_df, output_path)
    print(f"\nSaved figure: {output_path}")

    csv_path = args.output_dir / "model_comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()