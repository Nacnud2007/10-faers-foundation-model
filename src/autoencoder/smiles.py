from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_FILE = PROJECT_ROOT / "data" / "processed" / "drug_name_mapping_updated.csv"
PATIENT_FILE = PROJECT_ROOT / "data" / "processed" / "faers_combined_cleaned_pure_reactions.csv"
CACHE_FILE = PROJECT_ROOT / "cid_to_smiles_cache.csv"

X_OUTPUT_FILE = PROJECT_ROOT / "X_train_sparse.npz"
Y_OUTPUT_FILE = PROJECT_ROOT / "Y_train_sparse.npz"
ADR_VOCAB_FILE = PROJECT_ROOT / "adr_vocabulary.txt"
STATS_FILE = PROJECT_ROOT / "output" / "autoencoder" / "smiles_build_stats.json"
TEMP_DIR = PROJECT_ROOT / "output" / "autoencoder" / "smiles_tmp"

MAX_DRUGS = 5
MACCS_SIZE = 166
PUBCHEM_SIZE = 881
MORGAN_SIZE = 2048
FP_SIZE = MACCS_SIZE + PUBCHEM_SIZE + MORGAN_SIZE  # 3095
TOTAL_X_FEATURES = MAX_DRUGS * FP_SIZE
CHUNK_SIZE = 50,000 # RAM saving


def split_multi_value_field(raw_value: str | float | int | None) -> list[str]:
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    parts = re.split(r"[;,]", text)
    return [part.strip() for part in parts if part.strip() and part.strip().lower() not in {"nan", "none"}]


def detect_column(columns: list[str], preferred: str, keywords: tuple[str, ...]) -> str:
    if preferred in columns:
        return preferred

    lower_map = {column.lower(): column for column in columns}
    if preferred.lower() in lower_map:
        return lower_map[preferred.lower()]

    for column in columns:
        lower = column.lower()
        if any(keyword in lower for keyword in keywords):
            return column

    raise ValueError(f"Could not find a column matching {preferred!r} or keywords {keywords!r}.")


def load_cached_smiles(cache_file: Path) -> tuple[dict[int, str], dict[int, str]]:
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        return {}, {}

    print(f"Loading cached SMILES/fingerprints from local file: {cache_file}")
    cache_df = pd.read_csv(cache_file, dtype=str)
    cids = cache_df["cid"].astype(int)

    cid_to_smiles = dict(zip(cids, cache_df["smiles"]))

    if "pubchem_fp" in cache_df.columns:
        cid_to_pubchem_fp = dict(zip(cids, cache_df["pubchem_fp"]))
    else:
        cid_to_pubchem_fp = {}

    return cid_to_smiles, cid_to_pubchem_fp


def decode_pubchem_fingerprint(fingerprint_bits: str | float | None) -> np.ndarray:
    """Turn an 881-character '0'/'1' CACTVS fingerprint string into a uint8 array.

    Falls back to an all-zero block if the fingerprint is missing, blank, or the
    wrong length (e.g. NaN from an older cache file, or a failed PubChem lookup).
    """
    if not isinstance(fingerprint_bits, str) or len(fingerprint_bits) != PUBCHEM_SIZE:
        return np.zeros(PUBCHEM_SIZE, dtype=np.uint8)

    return np.array(list(fingerprint_bits), dtype=np.uint8)


def build_fingerprint_cache(
    mapping_df: pd.DataFrame,
    name_to_cid: dict[str, float],
    cid_to_smiles: dict[int, str],
    cid_to_pubchem_fp: dict[int, str],
) -> dict[str, np.ndarray]:
    print("Pre-computing active fingerprint bit positions for all unique drugs...")
    name_to_active_bits: dict[str, np.ndarray] = {}
    unique_names = mapping_df["messy_name"].dropna().unique()
    missing_pubchem_fp = 0

    for messy_name in unique_names:
        clean_name = str(messy_name).strip().lower()
        cid = name_to_cid.get(clean_name)
        active_bits = np.zeros(0, dtype=np.int32)

        if cid and not pd.isna(cid):
            cid_int = int(cid)
            smiles = cid_to_smiles.get(cid_int)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.uint8)[1:]
                    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_SIZE).GetFingerprintAsNumPy(mol).astype(np.uint8)

                    pubchem_bits = cid_to_pubchem_fp.get(cid_int)
                    pubchem = decode_pubchem_fingerprint(pubchem_bits)
                    if not pubchem.any() and not (isinstance(pubchem_bits, str) and len(pubchem_bits) == PUBCHEM_SIZE):
                        missing_pubchem_fp += 1

                    drug_vector = np.concatenate([maccs, pubchem, morgan])
                    active_bits = np.flatnonzero(drug_vector).astype(np.int32, copy=False)

        name_to_active_bits[clean_name] = active_bits

    if missing_pubchem_fp:
        print(f"   Warning: {missing_pubchem_fp} mapped drug(s) had no cached PubChem fingerprint (zero-filled).")

    name_to_active_bits["__pad__"] = np.zeros(0, dtype=np.int32)
    return name_to_active_bits


def first_pass(
    patient_file: Path,
    drug_column: str,
    target_column: str,
    name_to_active_bits: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    print(f"Scanning {patient_file} to build the full ADR vocabulary and sparse row counts...")

    x_count_chunks: list[np.ndarray] = []
    y_count_chunks: list[np.ndarray] = []
    unique_adrs: set[str] = set()
    total_rows = 0

    for chunk in pd.read_csv(
        patient_file,
        dtype=str,
        usecols=[drug_column, target_column],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        keep_default_na=False,
    ):
        chunk_x_counts: list[int] = []
        chunk_y_counts: list[int] = []

        for drug_value, adr_value in zip(chunk[drug_column], chunk[target_column]):
            raw_drugs = split_multi_value_field(drug_value)[:MAX_DRUGS]
            x_count = 0

            for drug_name in raw_drugs:
                active_bits = name_to_active_bits.get(drug_name.lower(), name_to_active_bits["__pad__"])
                x_count += int(active_bits.size)

            adr_values = list(dict.fromkeys(split_multi_value_field(adr_value)))
            unique_adrs.update(adr_values)

            chunk_x_counts.append(x_count)
            chunk_y_counts.append(len(adr_values))
            total_rows += 1

        x_count_chunks.append(np.asarray(chunk_x_counts, dtype=np.int32))
        y_count_chunks.append(np.asarray(chunk_y_counts, dtype=np.int32))

        print(f"   Counted {total_rows:,} rows so far...")

    x_row_counts = np.concatenate(x_count_chunks) if x_count_chunks else np.empty(0, dtype=np.int32)
    y_row_counts = np.concatenate(y_count_chunks) if y_count_chunks else np.empty(0, dtype=np.int32)
    return x_row_counts, y_row_counts, sorted(unique_adrs), total_rows


def counts_to_indptr(row_counts: np.ndarray) -> np.ndarray:
    indptr = np.empty(row_counts.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = np.cumsum(row_counts, dtype=np.int64)
    return indptr


def build_and_save_sparse_matrices(
    patient_file: Path,
    drug_column: str,
    target_column: str,
    adr_to_index: dict[str, int],
    name_to_active_bits: dict[str, np.ndarray],
    x_indptr: np.ndarray,
    y_indptr: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    total_rows = x_indptr.shape[0] - 1
    total_x_nnz = int(x_indptr[-1])
    total_y_nnz = int(y_indptr[-1])

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    x_indices_path = TEMP_DIR / "x_indices.dat"
    x_data_path = TEMP_DIR / "x_data.dat"
    y_indices_path = TEMP_DIR / "y_indices.dat"
    y_data_path = TEMP_DIR / "y_data.dat"

    print("Allocating disk-backed sparse buffers...")
    x_indices = np.memmap(x_indices_path, dtype=np.int32, mode="w+", shape=(total_x_nnz,))
    x_data = np.memmap(x_data_path, dtype=np.uint8, mode="w+", shape=(total_x_nnz,))
    y_indices = np.memmap(y_indices_path, dtype=np.int32, mode="w+", shape=(total_y_nnz,))
    y_data = np.memmap(y_data_path, dtype=np.uint8, mode="w+", shape=(total_y_nnz,))

    row_index = 0
    print("Filling sparse buffers from the raw CSV...")
    for chunk in pd.read_csv(
        patient_file,
        dtype=str,
        usecols=[drug_column, target_column],
        chunksize=CHUNK_SIZE,
        low_memory=False,
        keep_default_na=False,
    ):
        for drug_value, adr_value in zip(chunk[drug_column], chunk[target_column]):
            x_write_pos = int(x_indptr[row_index])
            y_write_pos = int(y_indptr[row_index])

            raw_drugs = split_multi_value_field(drug_value)[:MAX_DRUGS]
            for slot_index, drug_name in enumerate(raw_drugs):
                active_bits = name_to_active_bits.get(drug_name.lower(), name_to_active_bits["__pad__"])
                if active_bits.size:
                    slot_offset = slot_index * FP_SIZE
                    slot_bits = slot_offset + active_bits
                    next_pos = x_write_pos + slot_bits.size
                    x_indices[x_write_pos:next_pos] = slot_bits
                    x_data[x_write_pos:next_pos] = 1
                    x_write_pos = next_pos

            for adr in dict.fromkeys(split_multi_value_field(adr_value)):
                adr_index = adr_to_index.get(adr)
                if adr_index is not None:
                    y_indices[y_write_pos] = adr_index
                    y_data[y_write_pos] = 1
                    y_write_pos += 1

            row_index += 1

        print(f"   Filled {row_index:,} / {total_rows:,} rows...")

    x_matrix = sparse.csr_matrix((x_data, x_indices, x_indptr), shape=(total_rows, TOTAL_X_FEATURES), dtype=np.uint8)
    y_matrix = sparse.csr_matrix((y_data, y_indices, y_indptr), shape=(total_rows, len(adr_to_index)), dtype=np.uint8)

    print("Sorting indices for canonical CSR format...")
    x_matrix.sort_indices()
    y_matrix.sort_indices()
     
    print(f"Saving sparse features matrix to: {X_OUTPUT_FILE}")
    sparse.save_npz(X_OUTPUT_FILE, x_matrix)
    print(f"Saving sparse targets matrix to: {Y_OUTPUT_FILE}")
    sparse.save_npz(Y_OUTPUT_FILE, y_matrix)

    return x_matrix, y_matrix


def main() -> None:
    print("Loading mapping file...")
    mapping_df = pd.read_csv(MAPPING_FILE)
    name_to_cid = dict(zip(mapping_df["messy_name"].str.lower().str.strip(), mapping_df["pubchem_cid"]))
    cid_to_smiles, cid_to_pubchem_fp = load_cached_smiles(CACHE_FILE)

    def save_cache() -> None:
        all_cids = sorted(set(cid_to_smiles) | set(cid_to_pubchem_fp))
        pd.DataFrame(
            {
                "cid": all_cids,
                "smiles": [cid_to_smiles.get(cid, "") for cid in all_cids],
                "pubchem_fp": [cid_to_pubchem_fp.get(cid, "") for cid in all_cids],
            }
        ).to_csv(CACHE_FILE, index=False)

    unique_cids = mapping_df["pubchem_cid"].dropna().unique().astype(int).tolist()
    cids_to_fetch = [
        cid for cid in unique_cids if cid not in cid_to_smiles or cid not in cid_to_pubchem_fp
    ]

    if cids_to_fetch:
        print(f"Found {len(cids_to_fetch)} new CIDs to fetch from PubChem...")
        chunk_size = 20
        for i in range(0, len(cids_to_fetch), chunk_size):
            chunk = cids_to_fetch[i : i + chunk_size]
            try:
                compounds = pcp.get_compounds(chunk, "cid", timeout=10)
                for compound in compounds:
                    if compound.smiles:
                        cid_to_smiles[compound.cid] = compound.smiles
                    try:
                        fp = compound.cactvs_fingerprint
                    except Exception:
                        fp = None
                    if fp:
                        cid_to_pubchem_fp[compound.cid] = fp
                save_cache()
            except Exception:
                continue

    sample_df = pd.read_csv(PATIENT_FILE, nrows=1, keep_default_na=False)
    drug_column = detect_column(list(sample_df.columns), "drug_combination", ("drug", "combo"))
    target_column = detect_column(list(sample_df.columns), "adrs", ("adverse", "reaction", "event"))

    print(f"Using drug column:   {drug_column}")
    print(f"Using target column: {target_column}")

    name_to_active_bits = build_fingerprint_cache(mapping_df, name_to_cid, cid_to_smiles, cid_to_pubchem_fp)

    x_row_counts, y_row_counts, all_adrs, total_rows = first_pass(
        PATIENT_FILE,
        drug_column,
        target_column,
        name_to_active_bits,
    )

    adr_to_index = {adr: idx for idx, adr in enumerate(all_adrs)}
    print(f"Found {len(all_adrs):,} unique adverse events in the dataset.")
    ADR_VOCAB_FILE.write_text("\n".join(all_adrs))

    x_indptr = counts_to_indptr(x_row_counts)
    y_indptr = counts_to_indptr(y_row_counts)
    total_x_nnz = int(x_indptr[-1])
    total_y_nnz = int(y_indptr[-1])
    print(f"Total rows: {total_rows:,}")
    print(f"X nnz: {total_x_nnz:,}")
    print(f"Y nnz: {total_y_nnz:,}")

    x_matrix, y_matrix = build_and_save_sparse_matrices(
        PATIENT_FILE,
        drug_column,
        target_column,
        adr_to_index,
        name_to_active_bits,
        x_indptr,
        y_indptr,
    )

    stats = {
        "rows": total_rows,
        "x_shape": list(x_matrix.shape),
        "y_shape": list(y_matrix.shape),
        "x_nnz": total_x_nnz,
        "y_nnz": total_y_nnz,
        "adr_vocab_size": len(all_adrs),
        "x_output_file": str(X_OUTPUT_FILE),
        "y_output_file": str(Y_OUTPUT_FILE),
        "adr_vocab_file": str(ADR_VOCAB_FILE),
    }
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print("Cleaning up temporary buffers...")
    if TEMP_DIR.exists():
        for temp_file in TEMP_DIR.iterdir():
            try:
                temp_file.unlink()
            except OSError:
                pass
        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass

    print("\nSuccessfully completed the full sparse build.")


if __name__ == "__main__":
    main()