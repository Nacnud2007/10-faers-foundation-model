"""
Ccheck an existing multimodal_autoencoder_epoch.pt checkpoint.
Usage:
    python evaluate_checkpoint.py --checkpoint output/autoencoder/multimodal/multimodal_autoencoder_epoch13.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from autoencoder import (
    MultimodalAutoencoder,
    TrimodalDataset,
    BatchedTrimodalLoader,
    load_sparse_npz_fast,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
)
# Reused as-is -- these operate on plain (y_true, y_probs) arrays, so they work
# for any model's clin/ADR output, not just drug_adr_encoder.py's own model.
from drug_adr_encoder import sweep_thresholds, evaluate_by_frequency_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved multimodal autoencoder checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument(
        "--selected-adr-counts", type=Path,
        default=Path("output/autoencoder/drug_adr_encoder/selected_adr_counts.npy"),
        help="Per-column training counts saved by drug_adr_encoder.py, used for the "
             "rare/medium/common breakdown. Skipped if not found.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=10000,
                         help="Optional cap on rows to evaluate, for a quick look before running the full set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data...")
    chem_matrix = load_sparse_npz_fast(args.x_path)
    clin_matrix = load_sparse_npz_fast(args.y_path)
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")

    dataset = TrimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)

    indices = np.arange(len(dataset))
    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    print(f"Evaluating {len(indices):,} rows")

    loader = BatchedTrimodalLoader(dataset, batch_size=args.batch_size, shuffle=False, indices=indices)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False,
)
    print(f"  epoch: {checkpoint['epoch']}")
    # Older checkpoints (pre-split fix) saved a single 'losses' dict with train-only
    # numbers; newer ones save 'train_losses' and 'val_losses' separately.
    if "val_losses" in checkpoint:
        print(f"  val losses at save time: {checkpoint['val_losses']}")
    else:
        print(f"  losses at save time (train-only, no val split existed for this run): {checkpoint['losses']}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = MultimodalAutoencoder(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=checkpoint["args"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # critical -- turns off dropout, otherwise outputs are noisy and nondeterministic

    print(f"Running {len(loader):,} batches through the model...")
    all_clin_true, all_clin_probs = [], []
    all_trans_true, all_trans_pred = [], []

    with torch.no_grad():
        for batch_idx, (chem_batch, trans_batch, clin_batch) in enumerate(loader, start=1):
            chem_batch = chem_batch.to(device)
            trans_batch = trans_batch.to(device)
            clin_batch = clin_batch.to(device)

            trans_recon, clin_recon, _z = model(chem_batch, trans_batch, clin_batch)

            all_clin_true.append(clin_batch.cpu())
            all_clin_probs.append(torch.sigmoid(clin_recon).cpu())
            all_trans_true.append(trans_batch.cpu())
            all_trans_pred.append(trans_recon.cpu())

            if batch_idx % 200 == 0:
                print(f"  batch {batch_idx:,}/{len(loader):,}")

    clin_true = torch.cat(all_clin_true).numpy()
    clin_probs = torch.cat(all_clin_probs).numpy()
    trans_true = torch.cat(all_trans_true)
    trans_pred = torch.cat(all_trans_pred)

    # --- Clin head: precision/recall/F1 across thresholds, same logic as drug_adr_encoder.py ---
    print("\n=== Clinical (ADR) reconstruction ===")
    sweep_results, best = sweep_thresholds(clin_true, clin_probs)
    print(f"Best threshold: {best['threshold']:.2f} -> precision={best['precision']:.4f}, "
          f"recall={best['recall']:.4f}, F1={best['f1']:.4f}")
    print("(compare this F1 against 0.5-threshold F1 below -- a big gap means the model is fine, "
          "the 0.5 cutoff just wasn't right for how rare these ADRs are)")
    fixed_50 = next(r for r in sweep_results if abs(r["threshold"] - 0.5) < 1e-6)
    print(f"F1 at fixed 0.5 threshold: {fixed_50['f1']:.4f}")

    if args.selected_adr_counts.exists():
        selected_counts = np.load(args.selected_adr_counts)
        bucket_results = evaluate_by_frequency_bucket(
            clin_true, clin_probs, selected_counts, threshold=best["threshold"]
        )
        print("Per-frequency-bucket breakdown (rare/medium/common ADRs):")
        for bucket in bucket_results:
            print(f"  {bucket}")
    else:
        print(f"  (skipped frequency-bucket breakdown -- {args.selected_adr_counts} not found. "
              "Pass --selected-adr-counts if it's saved somewhere else.)")

    # --- Trans head: model MSE vs. trivial mean-baseline MSE ---
    print("\n=== Transcriptomic reconstruction ===")
    model_mse = torch.mean((trans_pred - trans_true) ** 2).item()
    baseline_pred = trans_true.mean(dim=0, keepdim=True).expand_as(trans_true)
    baseline_mse = torch.mean((baseline_pred - trans_true) ** 2).item()
    improvement = (baseline_mse - model_mse) / baseline_mse * 100 if baseline_mse > 0 else float("nan")
    print(f"Model MSE:          {model_mse:.6f}")
    print(f"Mean-baseline MSE:  {baseline_mse:.6f}")
    print(f"Improvement over baseline: {improvement:.1f}%")
    if improvement < 5:
        print("  ^ Model is barely beating 'always predict the average profile.' "
              "That's a sign it hasn't learned real per-drug transcriptomic signal yet.")

    print("\nReminder: these numbers are computed on data the model was trained on "
          "(no val split existed for this checkpoint). Treat this as a floor check, "
          "not a generalization estimate.")


if __name__ == "__main__":
    main()
