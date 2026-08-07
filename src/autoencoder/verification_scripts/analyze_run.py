"""
analyze_run.py

Diagnostic for MultimodalADRPredictor checkpoints (output/models/adr_predictor/).

Reuses the ACTUAL data loading, val-split, and batching logic from
autoencoder.py (BimodalDataset, BatchedBimodalLoader, PrefetchLoader,
split_indices) so evaluation matches training exactly -- no separate val
files, no reimplemented masking/scaling logic. The val split is
reconstructed from the checkpoint's own saved `args` (same
validation_fraction + split_seed used during training), so you're
evaluating on the same held-out rows training did.

For a given checkpoint:
  1. SATURATION PROBE: dead-unit check on chem_encoder / trans_encoder
     outputs, plus raw logit distribution, on one val batch.
  2. REAL METRICS: TP/FP/FN/TN threshold sweep + AUC-PR, pooled and
     per-ADR, restricted to the target ADR columns named in an
     --adr-indices CSV (resolved against --adr-vocab).

Run from the repo root:
  python src/autoencoder/analyze_run.py \
      --checkpoint output/models/adr_predictor/adr_predictor_epoch13.pt \
      --adr-indices data/processed/modified_top_100_adrs.csv \
      --adr-vocab adr_vocabulary.txt \
      --max-batches 50
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# autoencoder.py itself resolves PROJECT_ROOT via Path(__file__).parents[2],
# so its DEFAULT_* paths work regardless of cwd. This script's own sys.path
# gets src/autoencoder/ automatically (script's own dir) when run as
# `python src/autoencoder/analyze_run.py`, so these imports resolve the
# same way autoencoder.py's internal imports do.
from autoencoder import (  # noqa: E402
    MultimodalADRPredictor,
    BimodalDataset,
    BatchedBimodalLoader,
    PrefetchLoader,
    load_sparse_npz_fast,
    get_autocast_context,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
)
from drug_adr_encoder import split_indices  # noqa: E402


# ---------------------------------------------------------------------------
# ADR name/index resolution (unchanged approach: CSV of names -> vocab file)
# ---------------------------------------------------------------------------
def load_adr_indices(spec: str, vocab_path: str | None = None):
    """Returns (indices, names) in the order they appear in `spec`.

    - .npy file of raw ints -> used directly
    - comma-separated ints -> used directly
    - CSV with an 'ADR' name column -> looked up against adr_vocabulary.txt
      (line index == column index into clin_matrix / Y_train_sparse),
      REQUIRES --adr-vocab
    """
    if spec.endswith(".npy"):
        idx = np.load(spec).astype(int)
        return idx, [str(i) for i in idx]

    if spec.endswith(".csv"):
        if vocab_path is None:
            raise ValueError(
                "spec is a CSV of ADR names but --adr-vocab was not given. "
                "Pass --adr-vocab pointing at adr_vocabulary.txt."
            )
        vocab = [line.strip() for line in open(vocab_path, encoding="utf-8")]
        print(f"Loaded vocab with {len(vocab)} entries from {vocab_path}")
        name_to_idx = {name: i for i, name in enumerate(vocab)}

        names = []
        with open(spec, encoding="utf-8") as f:
            f.readline()  # skip header ("ADR,Frequency")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                names.append(line.split(",")[0].strip())

        indices, missing = [], []
        for name in names:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
            else:
                missing.append(name)

        if missing:
            raise ValueError(
                f"{len(missing)} of {len(names)} ADR names from {spec} were NOT "
                f"found in {vocab_path} (exact match). Unmatched: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        print(f"Matched all {len(names)} ADR names to vocabulary indices.")
        return np.array(indices, dtype=int), names

    idx = np.array([int(x) for x in spec.split(",")], dtype=int)
    return idx, [str(i) for i in idx]


# ---------------------------------------------------------------------------
# Part 1: Saturation probe
# ---------------------------------------------------------------------------
@dataclass
class SaturationReport:
    layer_stats: dict = field(default_factory=dict)
    logit_mean: float = 0.0
    logit_std: float = 0.0
    logit_min: float = 0.0
    logit_max: float = 0.0
    frac_logits_below_neg5: float = 0.0


def register_activation_hooks(targets: dict[str, nn.Module]):
    storage: dict[str, list[torch.Tensor]] = {name: [] for name in targets}

    def make_hook(name):
        def hook(_module, _inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            storage[name].append(out_t.detach().cpu())
        return hook

    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in targets.items()]
    return handles, storage


def summarize_dead_units(activations: torch.Tensor, eps: float = 1e-6) -> dict:
    per_unit_std = activations.reshape(-1, activations.shape[-1]).std(dim=0)
    dead_mask = per_unit_std < eps
    return {
        "n_units": activations.shape[-1],
        "n_dead": int(dead_mask.sum().item()),
        "frac_dead": float(dead_mask.float().mean().item()),
        "mean_cross_sample_std": float(per_unit_std.mean().item()),
        "max_cross_sample_std": float(per_unit_std.max().item()),
    }


def run_saturation_probe(model, chem_b, trans_b, trans_mask_b) -> SaturationReport:
    targets = {
        "chem_encoder": model.chem_encoder,
        "trans_encoder": model.trans_encoder,
        "fusion_layer": model.fusion_layer,
    }
    handles, storage = register_activation_hooks(targets)
    model.eval()
    with torch.no_grad():
        logits, _ = model(chem_b, trans_b, trans_mask_b)
    for h in handles:
        h.remove()

    report = SaturationReport()
    for name, chunks in storage.items():
        if not chunks:
            continue
        acts = torch.cat(chunks, dim=0)
        report.layer_stats[name] = summarize_dead_units(acts)

    logits_cpu = logits.detach().cpu().float()
    report.logit_mean = float(logits_cpu.mean().item())
    report.logit_std = float(logits_cpu.std().item())
    report.logit_min = float(logits_cpu.min().item())
    report.logit_max = float(logits_cpu.max().item())
    report.frac_logits_below_neg5 = float((logits_cpu < -5).float().mean().item())
    return report


def print_saturation_report(report: SaturationReport):
    print("\n=== SATURATION PROBE ===")
    print(f"Logits: mean={report.logit_mean:.4f} std={report.logit_std:.4f} "
          f"min={report.logit_min:.4f} max={report.logit_max:.4f}")
    print(f"Fraction of logits < -5 (sigmoid ~ 0, dead-gradient zone): "
          f"{report.frac_logits_below_neg5:.4f}")
    if report.logit_std < 0.05:
        print("  \u26a0\ufe0f  Logit std is near zero -- model outputs a near-constant "
              "prediction regardless of input. Collapse, not convergence.")
    for name, stats in report.layer_stats.items():
        print(f"\n[{name}] {stats['n_dead']}/{stats['n_units']} units dead "
              f"({stats['frac_dead']*100:.1f}%), "
              f"mean cross-sample std={stats['mean_cross_sample_std']:.6f}")
        if stats["frac_dead"] > 0.5:
            print(f"  \u26a0\ufe0f  Majority of {name} units are dead -- same pattern "
                  f"as the earlier trans_encoder collapse.")


# ---------------------------------------------------------------------------
# Part 2: Real metrics (TP/FP/FN/TN, threshold sweep, AUC-PR)
# ---------------------------------------------------------------------------
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    recall = tp_cum / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    results = []
    for t in thresholds:
        pred = y_prob >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": precision, "recall": recall, "f1": f1})
    return results


def print_metrics_report(probs: np.ndarray, labels: np.ndarray, adr_names=None):
    print("\n=== THRESHOLD SWEEP (aggregate across target ADRs) ===")
    thresholds = np.arange(0.05, 1.0, 0.05)
    y_true_flat = labels.flatten().astype(int)
    y_prob_flat = probs.flatten()
    results = threshold_sweep(y_true_flat, y_prob_flat, thresholds)
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        marker = "  <== best F1" if r is best else ""
        print(f"  t={r['threshold']:.2f}  TP={r['tp']:6d} FP={r['fp']:6d} "
              f"FN={r['fn']:6d} TN={r['tn']:8d}  "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}{marker}")

    print(f"\nOverall AUC-PR (target ADRs pooled): "
          f"{average_precision(y_true_flat, y_prob_flat):.4f}")
    print(f"Baseline (random-guess AUC-PR \u2248 positive rate): {y_true_flat.mean():.6f}")

    print("\n=== PER-ADR BREAKDOWN ===")
    n_adrs = labels.shape[1]
    per_adr = []
    for i in range(n_adrs):
        y_t = labels[:, i].astype(int)
        y_p = probs[:, i]
        ap = average_precision(y_t, y_p)
        support = int(y_t.sum())
        name = adr_names[i] if adr_names is not None else f"adr_{i}"
        per_adr.append((name, support, ap))

    per_adr.sort(key=lambda x: (x[2] if x[2] == x[2] else -1), reverse=True)
    print(f"{'ADR':<30} {'support':>8} {'AUC-PR':>8}")
    for name, support, ap in per_adr:
        ap_str = f"{ap:.4f}" if ap == ap else "n/a (0 pos)"
        print(f"{name:<30} {support:>8} {ap_str:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adr-indices", required=True,
                    help="CSV with an 'ADR' name column (needs --adr-vocab), "
                         "a .npy of ints, or comma-separated ints")
    p.add_argument("--adr-vocab", default=None,
                    help="path to adr_vocabulary.txt (line index = column "
                         "index into clin_matrix)")
    p.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    p.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    p.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    p.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=None,
                    help="cap eval batches for a quick smoke check")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    adr_indices, adr_names = load_adr_indices(args.adr_indices, vocab_path=args.adr_vocab)

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    validation_fraction = ckpt_args.get("validation_fraction", 0.1)
    split_seed = ckpt_args.get("split_seed", 0)
    trans_scale = ckpt_args.get("trans_scale", 1.0)
    print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"train_metrics={checkpoint.get('train_metrics')}, "
          f"val_metrics={checkpoint.get('val_metrics')}")
    print(f"Reconstructing val split with validation_fraction={validation_fraction}, "
          f"split_seed={split_seed} (from checkpoint's saved args)")

    print("Loading datasets (same as training)...")
    chem_matrix = load_sparse_npz_fast(args.x_path)
    clin_matrix = load_sparse_npz_fast(args.y_path)
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")

    non_zero_per_row = np.diff(chem_matrix.indptr)
    valid_indices = np.where(non_zero_per_row > 0)[0]

    train_rel, val_rel = split_indices(len(valid_indices), validation_fraction, split_seed)
    val_indices = valid_indices[val_rel]
    print(f"Val set: {len(val_indices):,} rows")

    dataset = BimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)
    val_loader = PrefetchLoader(
        BatchedBimodalLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            indices=val_indices, trans_scale=trans_scale,
            chemical_as_sparse=False,
        )
    )

    print("Building model...")
    model = MultimodalADRPredictor(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Saturation probe on one val batch ---
    chem_b, trans_b, trans_mask_b, _ = next(iter(val_loader))
    chem_b = chem_b.to(device)
    trans_b = trans_b.to(device)
    trans_mask_b = trans_mask_b.to(device)
    sat_report = run_saturation_probe(model, chem_b, trans_b, trans_mask_b)
    print_saturation_report(sat_report)

    # --- Full metrics pass over val set ---
    print("\nRunning full val evaluation for metrics...")
    all_logits, all_labels = [], []
    n_batches = len(val_loader)
    cap = args.max_batches or n_batches
    with torch.no_grad():
        for b, (chem_batch, trans_batch, trans_mask_batch, clin_batch) in enumerate(val_loader, start=1):
            if b > cap:
                break
            chem_batch = chem_batch.to(device)
            trans_batch = trans_batch.to(device)
            trans_mask_batch = trans_mask_batch.to(device)
            with get_autocast_context(device):
                logits, _ = model(chem_batch, trans_batch, trans_mask_batch)
            all_logits.append(logits.float().cpu().numpy()[:, adr_indices])
            all_labels.append(clin_batch.numpy()[:, adr_indices])
            if b % 20 == 0 or b == cap:
                print(f"  eval batch {b}/{cap}")

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    print_metrics_report(probs, labels, adr_names=adr_names)


if __name__ == "__main__":
    main()