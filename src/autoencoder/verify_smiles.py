"""
Sanity-check X_train_sparse.npz, Y_train_sparse.npz, adr_vocabulary.txt, and
smiles_build_stats.json against each other after running smiles.py.

Usage:
    python3 verify_build.py /Users/duncanpark/10-faers-foundation-model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

MACCS_SIZE = 166
PUBCHEM_SIZE = 881
MORGAN_SIZE = 2048
FP_SIZE = MACCS_SIZE + PUBCHEM_SIZE + MORGAN_SIZE  # 3_095
MAX_DRUGS = 5


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    x_path = root / "X_train_sparse.npz"
    y_path = root / "Y_train_sparse.npz"
    vocab_path = root / "adr_vocabulary.txt"
    stats_path = root / "output" / "autoencoder" / "smiles_build_stats.json"

    all_ok = True

    # --- Files exist and load without error ---
    for p in (x_path, y_path, vocab_path, stats_path):
        all_ok &= check(f"{p.name} exists", p.exists(), str(p))
    if not all_ok:
        print("\nStopping — missing files.")
        return

    try:
        X = sparse.load_npz(x_path)
    except Exception as e:
        all_ok &= check("X_train_sparse.npz loads", False, str(e))
        return
    all_ok &= check("X_train_sparse.npz loads", True)

    try:
        Y = sparse.load_npz(y_path)
    except Exception as e:
        all_ok &= check("Y_train_sparse.npz loads", False, str(e))
        return
    all_ok &= check("Y_train_sparse.npz loads", True)

    adr_vocab = vocab_path.read_text().splitlines()
    stats = json.loads(stats_path.read_text())

    print(f"\nX shape: {X.shape}, nnz: {X.nnz:,}")
    print(f"Y shape: {Y.shape}, nnz: {Y.nnz:,}")
    print(f"ADR vocab size: {len(adr_vocab):,}")
    print(f"stats.json: {json.dumps(stats, indent=2)}\n")

    # --- Cross-check shapes against stats.json ---
    all_ok &= check("X.shape matches stats.json", list(X.shape) == stats["x_shape"], f"{list(X.shape)} vs {stats['x_shape']}")
    all_ok &= check("Y.shape matches stats.json", list(Y.shape) == stats["y_shape"], f"{list(Y.shape)} vs {stats['y_shape']}")
    all_ok &= check("X.nnz matches stats.json", X.nnz == stats["x_nnz"], f"{X.nnz} vs {stats['x_nnz']}")
    all_ok &= check("Y.nnz matches stats.json", Y.nnz == stats["y_nnz"], f"{Y.nnz} vs {stats['y_nnz']}")
    all_ok &= check("row counts match (X vs Y vs stats)", X.shape[0] == Y.shape[0] == stats["rows"])
    all_ok &= check("X column count is MAX_DRUGS * FP_SIZE", X.shape[1] == MAX_DRUGS * FP_SIZE, f"{X.shape[1]} vs {MAX_DRUGS * FP_SIZE}")
    all_ok &= check("Y column count matches ADR vocab size", Y.shape[1] == len(adr_vocab))

    # --- Structural sanity ---
    all_ok &= check("X values are binary (0/1 only)", set(np.unique(X.data)).issubset({0, 1}))
    all_ok &= check("Y values are binary (0/1 only)", set(np.unique(Y.data)).issubset({0, 1}))
    all_ok &= check("X has sorted indices (valid CSR)", X.has_sorted_indices)
    all_ok &= check("Y has sorted indices (valid CSR)", Y.has_sorted_indices)
    x_check_copy = X.copy()
    x_check_copy.sum_duplicates()
    all_ok &= check("no duplicate column entries within any X row", x_check_copy.nnz == X.nnz, f"{x_check_copy.nnz} vs {X.nnz}")

    # --- Row-level checks (sampled, since 14.8M rows is too slow to check all densely) ---
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(X.shape[0], size=min(20_000, X.shape[0]), replace=False)
    x_sample = X[sample_idx]
    y_sample = Y[sample_idx]

    x_row_nnz = np.asarray(x_sample.sum(axis=1)).ravel()
    y_row_nnz = np.asarray(y_sample.sum(axis=1)).ravel()

    all_zero_x_rows = int((x_row_nnz == 0).sum())
    all_zero_y_rows = int((y_row_nnz == 0).sum())
    print(f"\nSampled {len(sample_idx):,} rows:")
    print(f"  X all-zero rows in sample: {all_zero_x_rows} ({100*all_zero_x_rows/len(sample_idx):.2f}%)")
    print(f"  Y all-zero rows in sample: {all_zero_y_rows} ({100*all_zero_y_rows/len(sample_idx):.2f}%)")
    all_ok &= check("most sampled X rows have at least one active bit", all_zero_x_rows < 0.5 * len(sample_idx))

    # --- Confirm the PubChem fingerprint fix actually took effect ---
    # For each of the 5 drug slots, check whether ANY nonzero bit falls in that
    # slot's PubChem sub-block (positions MACCS_SIZE : MACCS_SIZE+PUBCHEM_SIZE
    # within the slot). If pubchem were still all-zero (the old bug), none would.
    pubchem_hits = 0
    for slot in range(MAX_DRUGS):
        lo = slot * FP_SIZE + MACCS_SIZE
        hi = lo + PUBCHEM_SIZE
        block = x_sample[:, lo:hi]
        pubchem_hits += block.nnz

    print(f"\nTotal nonzero bits in PubChem sub-blocks across sample: {pubchem_hits:,}")
    all_ok &= check(
        "PubChem fingerprint block is populated (not the old all-zero bug)",
        pubchem_hits > 0,
        "if this is 0, the pubchem block is still zero-filled for every sampled row",
    )

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED — see [FAIL] lines above"))


if __name__ == "__main__":
    main()