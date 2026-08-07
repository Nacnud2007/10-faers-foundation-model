"""
Creates smaller, memory-safe versions of X_train_sparse.npz and Y_train_sparse.npz.

WHY THIS EXISTS:
  X_train_sparse.npz currently loads as ~52 GB in RAM because its index array
  (which stores WHERE each nonzero value lives) was saved as int64, when
  int32 is more than enough room for a 15,475-column matrix. On top of that,
  loading the file the normal way (`scipy.sparse.load_npz`) pulls in ALL
  14.8M rows even if you only plan to train on a subset of them.

WHAT THIS SCRIPT DOES:
  1. Picks a random subset of row indices (your --max-rows).
  2. Reads ONLY those rows' data off disk, using memory-mapping so the full
     46 GB index array never has to sit in RAM at once.
  3. Saves a new, much smaller .npz with int32 indices instead of int64.
  4. Does the same for Y_train_sparse.npz, using the exact same row indices,
     so X and Y stay aligned.

DISK SPACE NOTE:
  Unpacking X's raw arrays temporarily needs ~50-60 GB of free disk space
  (not RAM). Check with `df -h` before running.

USAGE:
  python subset_and_downcast.py --max-rows 500000

  Then point drug_adr_encoder.py at the new files:
  python drug_adr_encoder.py --x-path X_train_subset.npz --y-path Y_train_subset.npz --max-rows 500000
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse


def read_shape_only(npz_path: Path) -> tuple[int, int]:
    """Peeks at just the shape.npy entry inside the .npz zip, without extracting
    the huge data/indices arrays. Fast and cheap."""
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("shape.npy") as f:
            shape = np.load(io.BytesIO(f.read()))
    return int(shape[0]), int(shape[1])


def check_disk_space(npz_path: Path, min_free_gb: float = 60.0) -> None:
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    zip_size_gb = npz_path.stat().st_size / 1e9
    print(f"  {npz_path.name}: {zip_size_gb:.2f} GB compressed on disk, {free_gb:.1f} GB free")
    if free_gb < min_free_gb:
        print(
            f"  WARNING: less than {min_free_gb:.0f} GB free. Extraction may fail "
            "or fill your disk. Consider freeing space first."
        )


def extract_npz_arrays(npz_path: Path, extract_dir: Path) -> dict[str, Path]:
    """Unzips a scipy .npz sparse file so its component arrays can be mmap'd
    directly from disk instead of loaded fully into RAM."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        zf.extractall(extract_dir)
    return {
        "data": extract_dir / "data.npy",
        "indices": extract_dir / "indices.npy",
        "indptr": extract_dir / "indptr.npy",
        "shape": extract_dir / "shape.npy",
    }


def subset_csr_matrix(
    npz_path: Path,
    row_indices: np.ndarray,
    extract_dir: Path,
    index_dtype=np.int32,
) -> sparse.csr_matrix:
    """
    Builds a new CSR matrix containing only `row_indices`, reading only the
    needed slices of data/indices off disk (via mmap) rather than loading
    the entire arrays into RAM.
    """
    paths = extract_npz_arrays(npz_path, extract_dir)

    shape = tuple(np.load(paths["shape"]))
    indptr_full = np.load(paths["indptr"])  # small (~100MB), fine to load fully

    data_mmap = np.load(paths["data"], mmap_mode="r")
    indices_mmap = np.load(paths["indices"], mmap_mode="r")

    data_chunks = []
    indices_chunks = []
    new_indptr = np.zeros(len(row_indices) + 1, dtype=np.int64)

    for i, row in enumerate(row_indices):
        start, end = indptr_full[row], indptr_full[row + 1]
        data_chunks.append(np.array(data_mmap[start:end]))       # reads only this slice
        indices_chunks.append(np.array(indices_mmap[start:end]))  # reads only this slice
        new_indptr[i + 1] = new_indptr[i] + (end - start)

        if (i + 1) % 100_000 == 0:
            print(f"    processed {i + 1:,} / {len(row_indices):,} rows")

    new_data = np.concatenate(data_chunks)
    new_indices = np.concatenate(indices_chunks).astype(index_dtype)
    new_indptr = new_indptr.astype(np.int32)

    return sparse.csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=(len(row_indices), shape[1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x-path", type=Path, default=Path("X_train_sparse.npz"))
    parser.add_argument("--y-path", type=Path, default=Path("Y_train_sparse.npz"))
    parser.add_argument("--x-out", type=Path, default=Path("X_train_subset.npz"))
    parser.add_argument("--y-out", type=Path, default=Path("Y_train_subset.npz"))
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("Checking disk space...")
    check_disk_space(args.x_path)
    check_disk_space(args.y_path)

    total_rows, _ = read_shape_only(args.x_path)
    print(f"\nTotal rows in full dataset: {total_rows:,}")

    rng = np.random.default_rng(args.seed)
    row_indices = np.sort(rng.choice(total_rows, size=args.max_rows, replace=False))
    print(f"Selected {len(row_indices):,} rows (seed={args.seed})")

    print("\nBuilding X subset (this reads ~50GB off disk, may take a while)...")
    with tempfile.TemporaryDirectory() as tmp:
        x_subset = subset_csr_matrix(args.x_path, row_indices, Path(tmp) / "x")
    sparse.save_npz(args.x_out, x_subset)
    print(f"Saved {args.x_out} -- shape {x_subset.shape}, nnz {x_subset.nnz:,}, "
          f"indices dtype {x_subset.indices.dtype}")

    print("\nBuilding Y subset...")
    with tempfile.TemporaryDirectory() as tmp:
        y_subset = subset_csr_matrix(args.y_path, row_indices, Path(tmp) / "y")
    sparse.save_npz(args.y_out, y_subset)
    print(f"Saved {args.y_out} -- shape {y_subset.shape}, nnz {y_subset.nnz:,}, "
          f"indices dtype {y_subset.indices.dtype}")

    row_indices_path = args.x_out.with_suffix("").with_suffix(".row_indices.npy")
    np.save(row_indices_path, row_indices)
    print(f"\nSaved selected row indices to {row_indices_path}")
    print("\nDone. Update drug_adr_encoder.py's --x-path/--y-path to point at the subset files,")
    print("and note the subset is already row-limited, so --max-rows in that script")
    print("should be left at/above the subset size (it won't subset further).")


if __name__ == "__main__":
    main()