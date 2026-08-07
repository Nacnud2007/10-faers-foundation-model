"""Check whether the clinical encoder collapses by inspecting BCE weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_Y_PATH = PROJECT_ROOT / "Y_train_sparse.npz"


def load_sparse_npz_fast(path: Path) -> sparse.csr_matrix:
    """Fast, low-memory load for large scipy sparse matrices."""
    with np.load(path) as loaded:
        return sparse.csr_matrix(
            (loaded["data"], loaded["indices"], loaded["indptr"]),
            shape=tuple(loaded["shape"]),
            dtype=np.float32,
        )


def compute_clin_pos_weight(
    clin_matrix: sparse.csr_matrix, max_weight: float = 100.0
) -> torch.Tensor:
    """Computes capped positive weights for BCE loss to match training setup."""
    n_rows = clin_matrix.shape[0]
    pos_counts = np.asarray(clin_matrix.sum(axis=0)).ravel()
    pos_counts = np.clip(pos_counts, 1, None)
    neg_counts = n_rows - pos_counts
    pos_weight = np.clip(neg_counts / pos_counts, 1.0, max_weight)
    return torch.from_numpy(pos_weight.astype(np.float32))


def compute_baseline_bce_loss(
    y_sparse: sparse.csr_matrix,
    pos_weight: torch.Tensor,
    batch_size: int = 10_000,
) -> float:
    """
    Computes baseline BCE loss using background column frequencies,
    properly weighted by pos_weight and processed in chunks to avoid OOM.
    """
    num_samples, _ = y_sparse.shape

    # 1. Compute empirical positive prevalence per target column
    pos_counts = np.asarray(y_sparse.sum(axis=0)).ravel()
    pos_rates = pos_counts / float(num_samples)
    pos_rates = np.clip(pos_rates, 1e-7, 1.0 - 1e-7)

    # 2. Convert constant background probabilities into unscaled logits
    constant_logits = torch.from_numpy(
        np.log(pos_rates / (1.0 - pos_rates))
    ).float()

    # 3. Match reduction='mean' used in training BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    total_loss = 0.0
    total_batches = 0

    # Stream through rows in manageable chunks
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_y_dense = torch.from_numpy(
            y_sparse[start_idx:end_idx].toarray().astype(np.float32)
        )

        batch_logits = constant_logits.unsqueeze(0).expand(
            end_idx - start_idx, -1
        )

        with torch.no_grad():
            batch_loss = criterion(batch_logits, batch_y_dense)
            total_loss += batch_loss.item()
            total_batches += 1

    return total_loss / total_batches


def inspect_checkpoint(checkpoint_path: Path, y_path: Path, pos_weight_max: float) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    epoch = checkpoint.get("epoch", "unknown")
    train_metrics = checkpoint.get("train_metrics", {})
    val_metrics = checkpoint.get("val_metrics", {})

    print(f"=== Checkpoint: {checkpoint_path} (epoch {epoch}) ===")
    if train_metrics:
        print(f"Reported train loss: {train_metrics.get('loss', 0.0):.4f}")
    if val_metrics:
        print(f"Reported val loss:   {val_metrics.get('loss', 0.0):.4f}")

    print("\n--- LayerNorm gamma check ---")
    for key in ("chem_norm.weight", "trans_norm.weight"):
        if key in state_dict:
            weight = state_dict[key].numpy()
            abs_mean = np.mean(np.abs(weight))
            w_min = np.min(weight)
            w_max = np.max(weight)
            print(f"{key}: mean|gamma|={abs_mean:.4f}  min={w_min:.4f}  max={w_max:.4f}")
            if abs_mean < 0.05:
                print(f"  ⚠️  {key} looks collapsed (near-zero scale).")
        else:
            print(f"{key}: Not found in checkpoint.")

    print("\n--- Baseline comparison ---")
    if y_path.exists():
        print(f"Loading sparse Y matrix from {y_path.name}...")
        y_matrix = load_sparse_npz_fast(y_path)
        pos_weight = compute_clin_pos_weight(y_matrix, max_weight=pos_weight_max)

        print("Computing weighted baseline BCE loss...")
        baseline_loss = compute_baseline_bce_loss(y_matrix, pos_weight)
        print(f"Weighted Baseline BCE loss: {baseline_loss:.6f}")
    else:
        print(f"⚠️ Could not compute baseline: {y_path} does not exist.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect checkpoint weights and baseline loss.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH, help="Path to Y_train_sparse.npz")
    parser.add_argument("--pos-weight-max", type=float, default=100.0, help="Max weight clamp used during training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_checkpoint(args.checkpoint, args.y_path, args.pos_weight_max)


if __name__ == "__main__":
    main()
