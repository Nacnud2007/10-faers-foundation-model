"""
Create a compact visualization for the best multimodal ADR checkpoint.

This script is intentionally lightweight:
  - loads `output/models/adr_predictor/best_val.pt`
  - reconstructs the *exact* validation split from the checkpoint args,
    replicating main()'s preprocessing order in autoencoder.py:
      row-indices subset -> top-k ADR column selection -> qualifying-row
      filter -> max-rows subsample -> non-empty-chem filter -> split
  - evaluates on CPU by default
  - saves a two-panel figure similar to the screenshot:
      left  -> ADR frequency-bucket performance
      right -> precision/recall/F1 vs threshold

Example:
  python src/autoencoder/visualize_best_val_run.py
  python src/autoencoder/visualize_best_val_run.py --x-path X_train_subset.npz --y-path Y_train_subset.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autoencoder import (
    BimodalDataset,
    BatchedBimodalLoader,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    MultimodalADRPredictor,
    compute_clin_pos_weight,
    get_autocast_context,
    load_sparse_npz_fast,
)
from drug_adr_encoder import evaluate_by_frequency_bucket, split_indices, sweep_thresholds

DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "best_val.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "analysis"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "font.size": 11})


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


def load_subset_row_indices(x_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    # Skip auto-loading if x_path is already a subset file
    if "subset" in x_path.name.lower():
        return None
    candidate = x_path.with_suffix("").with_suffix(".row_indices.npy")
    return candidate if candidate.exists() else None

def build_val_loader(
    chem_matrix,
    clin_matrix,
    trans_profiles: np.ndarray,
    trans_patient_indices: np.ndarray,
    row_indices: np.ndarray | None,
    selected_adr_columns: np.ndarray,
    *,
    max_rows: int | None,
    validation_fraction: float,
    split_seed: int,
    trans_scale: float,
    batch_size: int,
):
    """Reproduce main()'s row/column filtering pipeline exactly, in order."""

    # 1. Optional row-indices subset (same as main()).
    if row_indices is not None:
        if chem_matrix.shape[0] != len(row_indices) or clin_matrix.shape[0] != len(row_indices):
            raise ValueError(
                "Subset row indices do not match X/Y row counts. "
                "Make sure the row_indices file came from the same subset."
            )
        chem_matrix = chem_matrix[row_indices]
        clin_matrix = clin_matrix[row_indices]
        trans_patient_indices = trans_patient_indices[row_indices]

    # 2. Top-k ADR column selection -- reuse the exact columns saved in the
    #    checkpoint rather than recomputing.
    clin_matrix = clin_matrix[:, selected_adr_columns].tocsr()

    # 3. Qualifying-row filter: at least one selected ADR present.
    full_target_rows = np.asarray(clin_matrix.sum(axis=1)).ravel() > 0
    qualifying_rows = np.flatnonzero(full_target_rows)

    # 4. Max-rows subsample, reproduced with the same seeded RNG call as main().
    if max_rows is not None:
        if max_rows < len(qualifying_rows):
            rng = np.random.default_rng(split_seed)
            selected_row_indices = rng.choice(qualifying_rows, size=max_rows, replace=False)
            selected_row_indices.sort()
        else:
            selected_row_indices = qualifying_rows
    else:
        selected_row_indices = qualifying_rows

    chem_matrix = chem_matrix[selected_row_indices]
    clin_matrix = clin_matrix[selected_row_indices]
    trans_patient_indices = trans_patient_indices[selected_row_indices]

    # 5. Non-empty chemical vector filter (same as main()).
    non_zero_per_row = np.diff(chem_matrix.indptr)
    valid_indices = np.where(non_zero_per_row > 0)[0]

    # 6. Same train/val split call, over the already-filtered row set.
    train_rel, val_rel = split_indices(len(valid_indices), validation_fraction, split_seed)
    train_indices = valid_indices[train_rel]
    val_indices = valid_indices[val_rel]

    dataset = BimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)
    loader = BatchedBimodalLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        indices=val_indices,
        trans_scale=trans_scale,
        chemical_as_sparse=True,
    )
    return train_indices, val_indices, loader, clin_matrix


def evaluate_checkpoint(
    checkpoint_path: Path,
    x_path: Path,
    y_path: Path,
    trans_profiles_path: Path,
    trans_patient_indices_path: Path,
    row_indices_path: Path | None,
    *,
    batch_size: int,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    validation_fraction = float(ckpt_args.get("validation_fraction", 0.1))
    split_seed = int(ckpt_args.get("split_seed", 0))
    trans_scale = float(ckpt_args.get("trans_scale", 1.0))
    dropout = float(ckpt_args.get("dropout", 0.0))
    pos_weight_max = float(ckpt_args.get("clin_pos_weight_max", 20.0))
    max_rows = ckpt_args.get("max_rows", None)

    selected_adr_columns = checkpoint["selected_adr_columns"]
    print(f"Checkpoint trained on {len(selected_adr_columns):,} selected ADR columns.")
    if max_rows is not None:
        print(f"Checkpoint trained with --max-rows {max_rows:,}.")

    chem_matrix = load_sparse_npz_fast(x_path)
    clin_matrix = load_sparse_npz_fast(y_path)
    trans_profiles = np.load(trans_profiles_path, mmap_mode="r")
    trans_patient_indices = np.load(trans_patient_indices_path, mmap_mode="r")
    row_indices = np.load(row_indices_path, mmap_mode="r") if row_indices_path is not None else None

    train_indices, val_indices, val_loader, clin_matrix = build_val_loader(
        chem_matrix,
        clin_matrix,
        trans_profiles,
        trans_patient_indices,
        row_indices,
        selected_adr_columns,
        max_rows=max_rows,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        trans_scale=trans_scale,
        batch_size=batch_size,
    )

    # train_counts is now computed post-column-selection/post-row-filtering,
    # so it lines up column-for-column with y_true/y_prob below.
    train_counts = np.asarray(clin_matrix[train_indices].sum(axis=0)).ravel().astype(np.int64, copy=False)

    # Match training exactly: pos_weight is computed over the full filtered
    # clin_matrix (train + val), not train-only.
    pos_weight = compute_clin_pos_weight(clin_matrix, max_weight=pos_weight_max).to(device)

    model = MultimodalADRPredictor(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_chunks = []
    y_prob_chunks = []
    losses = []
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    with torch.inference_mode():
        for chem_batch, trans_batch, trans_mask_batch, clin_batch in val_loader:
            if isinstance(chem_batch, torch.Tensor):
                chem_batch = chem_batch.to(device)
            trans_batch = trans_batch.to(device)
            trans_mask_batch = trans_mask_batch.to(device)
            clin_batch = clin_batch.to(device)

            with get_autocast_context(device):
                logits, _ = model(chem_batch, trans_batch, trans_mask_batch)
                loss = criterion(logits.float(), clin_batch.float())

            losses.append(float(loss.item()))
            y_true_chunks.append(clin_batch.float().cpu().numpy())
            y_prob_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

    y_true = np.vstack(y_true_chunks)
    y_prob = np.vstack(y_prob_chunks)
    y_true_flat = y_true.reshape(-1).astype(np.int32)
    y_prob_flat = y_prob.reshape(-1)
    sweep_rows, best = sweep_thresholds(y_true_flat, y_prob_flat)
    bucket_results = evaluate_by_frequency_bucket(y_true, y_prob, train_counts, float(best["threshold"]))

    return {
        "checkpoint": checkpoint,
        "val_indices": val_indices,
        "loss": float(np.mean(losses)),
        "y_true": y_true,
        "y_prob": y_prob,
        "sweep_rows": sweep_rows,
        "best": best,
        "bucket_results": bucket_results,
    }


def make_figure(bucket_results: list[dict], sweep_rows: list[dict], best: dict, output_path: Path) -> None:
    bucket_df = pd.DataFrame(bucket_results)
    sweep_df = pd.DataFrame(sweep_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not bucket_df.empty:
        def bucket_metric_error(score: float, support: int) -> float:
            """Approximate a standard error for display purposes."""
            support = max(int(support), 1)
            score = float(np.clip(score, 0.0, 1.0))
            return float(np.sqrt(max(score * (1.0 - score), 1e-6) / support))

        melted = bucket_df.melt(
            id_vars=["bucket"],
            value_vars=["precision", "recall", "f1", "auc_pr"],
            var_name="Metric",
            value_name="Score",
        )
        sns.barplot(data=melted, x="bucket", y="Score", hue="Metric", ax=axes[0], palette="Blues")
        axes[0].set_title("Performance by ADR Frequency Bucket", fontweight="bold")
        axes[0].set_xlabel("ADR Frequency Bucket")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0.0, 1.05)

        metric_order = ["PRECISION", "RECALL", "F1", "AUC_PR"]
        for container, metric in zip(axes[0].containers, metric_order):
            for idx, bar in enumerate(container):
                if idx >= len(bucket_df):
                    continue
                row = bucket_df.iloc[idx]
                support = int(row.get("n_columns", 1))
                score = float(bar.get_height())
                if score <= 0:
                    continue
                err = bucket_metric_error(score, support)
                axes[0].errorbar(
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
    else:
        axes[0].text(0.5, 0.5, "No bucket results available", ha="center", va="center")
        axes[0].set_axis_off()

    axes[1].plot(sweep_df["threshold"], sweep_df["precision"], color="#1f3b73", lw=2, label="Precision")
    axes[1].plot(sweep_df["threshold"], sweep_df["recall"], color="#ff4d4d", lw=2, label="Recall")
    axes[1].plot(sweep_df["threshold"], sweep_df["f1"], color="#2a9d8f", lw=2, linestyle="--", label="F1 Score")
    axes[1].axvline(float(best["threshold"]), color="black", linestyle=":", lw=1.5, label=f"Best Threshold ({best['threshold']:.2f})")
    axes[1].set_title("Precision-Recall-F1 Trade-off across Decision Thresholds", fontweight="bold")
    axes[1].set_xlabel("Decision Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the best ADR predictor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument("--row-indices", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    device = resolve_device(args.device)
    row_indices_path = load_subset_row_indices(args.x_path, args.row_indices)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    if row_indices_path is not None:
        print(f"Using subset row indices: {row_indices_path}")

    result = evaluate_checkpoint(
        args.checkpoint,
        args.x_path,
        args.y_path,
        args.trans_profiles,
        args.trans_patient_indices,
        row_indices_path,
        batch_size=args.batch_size,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "best_val_visualization.png"
    make_figure(result["bucket_results"], result["sweep_rows"], result["best"], output_path)

    print("\n=== Best Checkpoint Summary ===")
    print(f"Rows evaluated: {len(result['val_indices']):,}")
    print(f"Mean BCE loss:   {result['loss']:.6f}")
    print(f"Best threshold:  {float(result['best']['threshold']):.2f}")
    print(f"Precision:       {float(result['best']['precision']):.4f}")
    print(f"Recall:          {float(result['best']['recall']):.4f}")
    print(f"F1:              {float(result['best']['f1']):.4f}")
    print(f"Saved figure:    {output_path}")


if __name__ == "__main__":
    main()
