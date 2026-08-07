"""
Print validation metrics for a single multimodal ADR checkpoint.

This is the lightweight version of the run analysis:
  - loads one checkpoint, defaulting to `best_val.pt`
  - runs on CPU by default to avoid MPS unified-memory spikes
  - reconstructs the *exact* validation split from the saved training args,
    replicating main()'s preprocessing order in autoencoder.py:
      row-indices subset -> top-k ADR column selection -> qualifying-row
      filter -> max-rows subsample -> non-empty-chem filter -> split
  - prints precision, recall, F1, AP, ROC-AUC, Brier score, and a
    threshold sweep to find the best F1 cutoff

Example:
  python src/autoencoder/report_best_val_metrics.py
  python src/autoencoder/report_best_val_metrics.py --checkpoint output/models/adr_predictor/best_val.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from drug_adr_encoder import split_indices, sweep_thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "best_val.pt"


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
                "Make sure the subset row_indices file was generated with the same subset files."
            )
        chem_matrix = chem_matrix[row_indices]
        clin_matrix = clin_matrix[row_indices]
        trans_patient_indices = trans_patient_indices[row_indices]

    # 2. Top-k ADR column selection -- reuse the exact columns saved in the
    #    checkpoint rather than recomputing (avoids depending on
    #    choose_top_adverse_event_columns being deterministic across runs).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Print metrics for best_val.pt.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument(
        "--row-indices",
        type=Path,
        default=None,
        help="Optional row-index file matching a subset X/Y pair, usually X_train_subset.row_indices.npy.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional quick sanity cap.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    chem_matrix = load_sparse_npz_fast(args.x_path)
    clin_matrix = load_sparse_npz_fast(args.y_path)
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")
    inferred_row_indices = args.row_indices
    if inferred_row_indices is None:
        candidate = args.x_path.with_suffix("").with_suffix(".row_indices.npy")
        if candidate.exists():
            inferred_row_indices = candidate
    row_indices = np.load(inferred_row_indices, mmap_mode="r") if inferred_row_indices is not None else None

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
        batch_size=args.batch_size,
    )

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
        for batch_idx, (chem_batch, trans_batch, trans_mask_batch, clin_batch) in enumerate(val_loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break

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
    best_threshold = float(best["threshold"])
    y_pred_flat = (y_prob_flat > best_threshold).astype(np.int32)

    tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))
    fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
    fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))
    tn = int(np.sum((y_true_flat == 0) & (y_pred_flat == 0)))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])

    print("\n=== Validation Summary ===")
    print(f"Rows evaluated: {len(val_indices):,}")
    print(f"ADR columns evaluated: {clin_matrix.shape[1]:,}")
    print(f"Mean BCE loss:   {float(np.mean(losses)):.6f}")
    print(f"Positive rate:   true={y_true_flat.mean():.6f} pred={y_pred_flat.mean():.6f}")
    print(f"Best threshold:  {best_threshold:.2f}")
    print(f"Precision:       {precision_score(y_true_flat, y_pred_flat, zero_division=0):.4f}")
    print(f"Recall:          {recall_score(y_true_flat, y_pred_flat, zero_division=0):.4f}")
    print(f"F1:              {f1_score(y_true_flat, y_pred_flat, zero_division=0):.4f}")
    print(f"Accuracy:        {accuracy_score(y_true_flat, y_pred_flat):.4f}")
    print(f"Average precision (PR-AUC): {average_precision_score(y_true_flat, y_prob_flat):.4f}")
    try:
        print(f"ROC AUC:         {roc_auc_score(y_true_flat, y_prob_flat):.4f}")
    except ValueError:
        print("ROC AUC:         n/a (only one class present)")
    print(f"Brier score:     {brier_score_loss(y_true_flat, y_prob_flat):.6f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Confusion matrix array:\n{cm}")

    print("\n=== Threshold Sweep ===")
    for row in sweep_rows:
        marker = "  <== best" if abs(row["threshold"] - best_threshold) < 1e-9 else ""
        print(
            f"t={row['threshold']:.2f}  "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}{marker}"
        )


if __name__ == "__main__":
    main()