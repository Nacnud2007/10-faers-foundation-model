"""
Evaluation script for the MultimodalADRPredictor.

Loads a trained checkpoint, runs inference over a subset of the held-out
validation split (same split logic/seed as training), and reports:

  - Per-label Average Precision (AUPRC), macro-averaged over labels with
    at least one positive in the evaluated subset. This is the primary
    metric under this level of class imbalance -- accuracy/threshold
    metrics are close to meaningless when true positive rate is ~0.015%.
  - The macro-averaged trivial baseline AUPRC: always predicting each
    label's *training-set* prevalence for every sample. The AP of a
    constant predictor equals prevalence, so this is a free, exact
    "did the model learn anything beyond label frequency" reference.
  - Per-label AUROC, macro-averaged (secondary; also imbalance-sensitive
    but a common companion metric).
  - A prediction-diversity check: mean cosine similarity between
    predicted probability vectors (and bottleneck z embeddings) for
    randomly paired, *different* drugs. Values near 1.0 mean the model
    is producing nearly the same output regardless of input -- i.e. it
    has collapsed onto the population marginal rather than learning
    drug-specific signal.

Usage:
    python evaluate_adr_predictor.py --checkpoint output/models/adr_predictor/best_val.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from autoencoder import (
    BatchedBimodalLoader,
    BimodalDataset,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_TRANS_SCALE,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    MultimodalADRPredictor,
    PrefetchLoader,
    load_sparse_npz_fast,
)
from drug_adr_encoder import split_indices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "output" / "models" / "adr_predictor" / "best_val.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ADR predictor checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0,
                         help="Must match the --split-seed used during training so this "
                              "rebuilds the same held-out indices, not a leaked training subset.")
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=10_000,
        help="Random subset of the validation split to evaluate. The full val split (>1M rows) "
             "x thousands of labels as dense float arrays won't comfortably fit in memory, so "
             "this subsamples for a fast, still-representative estimate.",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--trans-scale", type=float, default=DEFAULT_TRANS_SCALE)
    parser.add_argument("--min-label-positives", type=int, default=1,
                         help="Skip labels with fewer than this many positives in the evaluated "
                              "subset -- AUPRC/AUROC are undefined or meaningless with zero positives.")
    return parser.parse_args()


def rebuild_val_indices(chem_matrix, args: argparse.Namespace) -> np.ndarray:
    """Reproduces the exact train/val split from training so eval only ever sees held-out rows."""
    non_zero_per_row = np.diff(chem_matrix.indptr)
    valid_indices = np.where(non_zero_per_row > 0)[0]
    _, val_rel = split_indices(len(valid_indices), args.validation_fraction, args.split_seed)
    return valid_indices[val_rel]


def mean_cosine_sim(mat: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> float:
    a, b = mat[a_idx], mat[b_idx]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a_norm * b_norm).sum(axis=1).mean())


def main() -> None:
    args = parse_args()

    print("Loading data...")
    chem_matrix = load_sparse_npz_fast(args.x_path)
    clin_matrix = load_sparse_npz_fast(args.y_path)
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")

    val_indices = rebuild_val_indices(chem_matrix, args)
    rng = np.random.default_rng(args.sample_seed)
    if len(val_indices) > args.eval_samples:
        val_indices = rng.choice(val_indices, size=args.eval_samples, replace=False)
    print(f"Evaluating on {len(val_indices):,} held-out samples "
          f"(sample-seed={args.sample_seed}, split-seed={args.split_seed})")

    dataset = BimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)
    loader = PrefetchLoader(
        BatchedBimodalLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            indices=val_indices, trans_scale=args.trans_scale,
        )
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalADRPredictor(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels, all_z = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for chem_batch, trans_batch, clin_batch in loader:
            chem_batch = chem_batch.to(device)
            trans_batch = trans_batch.to(device)
            logits, z = model(chem_batch, trans_batch)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(clin_batch.numpy())
            all_z.append(z.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    z = np.concatenate(all_z, axis=0)
    print(f"Collected predictions: probs={probs.shape}, labels={labels.shape}")

    # --- Per-label AUPRC / AUROC, macro-averaged over labels with enough positives ---
    train_prevalence = np.asarray(clin_matrix.sum(axis=0)).ravel() / clin_matrix.shape[0]

    label_pos_counts = labels.sum(axis=0)
    eval_label_mask = label_pos_counts >= args.min_label_positives
    n_eval_labels = int(eval_label_mask.sum())
    print(f"\nEvaluating {n_eval_labels:,} / {labels.shape[1]:,} labels "
          f"with >= {args.min_label_positives} positive(s) in this subset")

    aps, aurocs, baseline_aps = [], [], []
    for label_idx in np.where(eval_label_mask)[0]:
        y_true = labels[:, label_idx]
        y_score = probs[:, label_idx]
        aps.append(average_precision_score(y_true, y_score))
        baseline_aps.append(train_prevalence[label_idx])  # AP of a constant predictor == prevalence
        if y_true.min() != y_true.max():  # AUROC undefined with only one class present
            aurocs.append(roc_auc_score(y_true, y_score))

    macro_ap = float(np.mean(aps))
    macro_baseline_ap = float(np.mean(baseline_aps))
    macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

    print(f"\nMacro AUPRC (model):                {macro_ap:.4f}")
    print(f"Macro AUPRC (prevalence baseline):  {macro_baseline_ap:.4f}")
    print(f"Macro AUPRC skill vs. baseline:     {macro_ap - macro_baseline_ap:+.4f}")
    print(f"Macro AUROC (model):                {macro_auroc:.4f}  (n={len(aurocs):,} labels)")

    # --- Prediction-diversity check: are different drugs getting different predictions? ---
    n_pairs = min(2000, probs.shape[0] // 2)
    idx_a = rng.choice(probs.shape[0], size=n_pairs, replace=False)
    idx_b = rng.choice(probs.shape[0], size=n_pairs, replace=False)
    different_mask = idx_a != idx_b
    idx_a, idx_b = idx_a[different_mask], idx_b[different_mask]

    prob_sim = mean_cosine_sim(probs, idx_a, idx_b)
    z_sim = mean_cosine_sim(z, idx_a, idx_b)

    print("\nMean cosine similarity between random *different*-drug pairs:")
    print(f"  Predicted probability vectors: {prob_sim:.4f}")
    print(f"  Bottleneck z embeddings:       {z_sim:.4f}")
    print("  (Values near 1.0 suggest the model is largely ignoring per-drug input and")
    print("   predicting close to the population-level label distribution for everyone.)")


if __name__ == "__main__":
    main()