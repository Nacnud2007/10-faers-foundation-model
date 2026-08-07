"""Prepare transcriptomic training data and mappings for the autoencoder."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PATIENT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "faers_combined_cleaned_pure_reactions.csv"
)
DEFAULT_TRANSCRIPTOMIC_FILE = PROJECT_ROOT / "pseudobulk_perturb.csv"
DEFAULT_MAPPING_FILE = (
    PROJECT_ROOT / "data" / "processed" / "drug_name_mapping_updated.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "transcriptomic"

DEFAULT_MAX_DRUGS = 5
DEFAULT_PATIENT_CHUNK_SIZE = 50_000
DEFAULT_TRANSCRIPTOMIC_CHUNK_SIZE = 512
HUGE_MATERIALIZED_NNZ_LIMIT = 100_000_000

DRUG_SPLIT_RE = re.compile(r"[;,|]")
NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_EXCLUDED_DRUG_TERMS = (
    "proactiv",
    "face wash",
    "cleanser",
    "cleansing wash",
    "wash",
    "toner",
    "moisturizer",
    "moisturiser",
    "spf",
    "sunscreen",
    "mask",
    "serum",
    "cosmetic",
    "blemish",
    "blackhead",
    "exfoliator",
)


@dataclass(frozen=True)
class CompactPaths:
    output_dir: Path
    prefix: str

    @property
    def profiles(self) -> Path:
        return self.output_dir / f"{self.prefix}_drug_profiles.npy"

    @property
    def patient_indices(self) -> Path:
        return self.output_dir / f"{self.prefix}_patient_profile_indices.npy"

    @property
    def profile_names(self) -> Path:
        return self.output_dir / f"{self.prefix}_profile_drugs.txt"

    @property
    def gene_names(self) -> Path:
        return self.output_dir / f"{self.prefix}_genes.txt"

    @property
    def alias_map(self) -> Path:
        return self.output_dir / f"{self.prefix}_aliases.csv"

    @property
    def stats(self) -> Path:
        return self.output_dir / f"{self.prefix}_stats.json"

    @property
    def tmp_profile_blocks(self) -> Path:
        return self.output_dir / f".{self.prefix}_profile_blocks.tmp.npy"

    @property
    def tmp_patient_blocks(self) -> Path:
        return self.output_dir / f".{self.prefix}_patient_index_blocks.tmp.npy"


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""

    cleaned = NAME_CLEAN_RE.sub(" ", str(value).strip().lower())
    return " ".join(cleaned.split())


def build_excluded_terms(
    extra_terms: Iterable[str] = (),
    *,
    include_defaults: bool = True,
) -> tuple[str, ...]:
    terms = set()
    if include_defaults:
        terms.update(DEFAULT_EXCLUDED_DRUG_TERMS)
    terms.update(extra_terms)

    return tuple(
        sorted(
            normalize_name(term)
            for term in terms
            if normalize_name(term)
        )
    )


def is_excluded_drug(drug_name: str, excluded_terms: tuple[str, ...]) -> bool:
    return any(term in drug_name for term in excluded_terms)


def parse_drug_list(
    value: object,
    max_drugs: int,
    excluded_terms: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    if pd.isna(value):
        return [], []

    drugs: list[str] = []
    excluded_drugs: list[str] = []
    for token in DRUG_SPLIT_RE.split(str(value)):
        drug = normalize_name(token)
        if not drug:
            continue
        if is_excluded_drug(drug, excluded_terms):
            excluded_drugs.append(drug)
            continue
        drugs.append(drug)

    return drugs, excluded_drugs


def csv_separator(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def detect_column(
    csv_path: Path,
    preferred_names: Iterable[str],
    keyword_fallbacks: Iterable[str],
    *,
    sep: str = ",",
    allow_first_column: bool = False,
) -> str:
    columns = pd.read_csv(csv_path, sep=sep, nrows=0).columns.tolist()
    lower_to_original = {column.lower(): column for column in columns}

    for preferred_name in preferred_names:
        match = lower_to_original.get(preferred_name.lower())
        if match is not None:
            return match

    for keyword in keyword_fallbacks:
        matches = [column for column in columns if keyword.lower() in column.lower()]
        if matches:
            return matches[0]

    if allow_first_column and columns:
        return columns[0]

    raise ValueError(
        f"Could not detect a suitable column in {csv_path}. "
        f"Available columns include: {columns[:12]}"
    )


def load_name_map(mapping_file: Path) -> dict[str, str]:
    mapping_df = pd.read_csv(
        mapping_file,
        usecols=["messy_name", "standardized_name"],
        dtype=str,
    )

    name_map: dict[str, str] = {}
    for messy_name, standardized_name in zip(
        mapping_df["messy_name"], mapping_df["standardized_name"]
    ):
        messy_key = normalize_name(messy_name)
        standardized_key = normalize_name(standardized_name)

        if messy_key and standardized_key:
            name_map[messy_key] = standardized_key
        if standardized_key:
            name_map.setdefault(standardized_key, standardized_key)

    return name_map


def resolve_drug_name(drug_name: str, name_map: dict[str, str]) -> str:
    return name_map.get(drug_name, drug_name)


def remove_stale_outputs(paths: CompactPaths, csr_output_file: Path | None) -> None:
    targets = [
        paths.profiles,
        paths.patient_indices,
        paths.profile_names,
        paths.gene_names,
        paths.alias_map,
        paths.stats,
        paths.tmp_profile_blocks,
        paths.tmp_patient_blocks,
    ]

    if csr_output_file is not None:
        targets.append(csr_output_file)

    for target in targets:
        target.unlink(missing_ok=True)


def iter_appended_npy(path: Path) -> Iterator[np.ndarray]:
    with path.open("rb") as handle:
        while True:
            try:
                yield np.load(handle)
            except (EOFError, ValueError):
                break


def write_lines(path: Path, values: Iterable[str]) -> None:
    with path.open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def compact_appended_arrays(
    appended_path: Path,
    final_path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    output = np.lib.format.open_memmap(
        final_path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )

    cursor = 0
    for block in iter_appended_npy(appended_path):
        next_cursor = cursor + len(block)
        output[cursor:next_cursor] = block
        cursor = next_cursor

    output.flush()
    del output

    if cursor != shape[0]:
        raise RuntimeError(
            f"Expected {shape[0]:,} rows while compacting {appended_path}, "
            f"but wrote {cursor:,}."
        )


def load_transcriptomic_profiles(
    transcriptomic_file: Path,
    transcriptomic_drug_column: str,
    name_map: dict[str, str],
    paths: CompactPaths,
    *,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[dict[str, int], list[str], list[str]]:
    sep = csv_separator(transcriptomic_file)
    header = pd.read_csv(transcriptomic_file, sep=sep, nrows=0).columns.tolist()
    gene_columns = [column for column in header if column != transcriptomic_drug_column]

    if not gene_columns:
        raise ValueError(f"No gene columns found in {transcriptomic_file}.")

    print(
        f"Loading transcriptomic profile bank from {transcriptomic_file} "
        f"({len(gene_columns):,} genes)..."
    )

    dtype_map = {column: np.float32 for column in gene_columns}
    dtype_map[transcriptomic_drug_column] = str

    profile_names: list[str] = []
    name_to_profile_index: dict[str, int] = {}
    rows_seen = 0
    duplicate_names = 0

    read_csv_kwargs = {
        "sep": sep,
        "usecols": [transcriptomic_drug_column, *gene_columns],
        "dtype": dtype_map,
        "chunksize": chunk_size,
        "low_memory": False,
    }
    if max_rows is not None:
        read_csv_kwargs["nrows"] = max_rows

    with paths.tmp_profile_blocks.open("ab") as tmp_profiles:
        for chunk in pd.read_csv(transcriptomic_file, **read_csv_kwargs):
            profile_matrix = chunk[gene_columns].to_numpy(dtype=np.float32, copy=False)
            selected_rows: list[int] = []

            for local_row, raw_value in enumerate(chunk[transcriptomic_drug_column]):
                raw_key = normalize_name(raw_value)
                if not raw_key:
                    continue

                canonical_key = resolve_drug_name(raw_key, name_map)
                profile_index = name_to_profile_index.get(canonical_key)

                if profile_index is None:
                    profile_index = len(profile_names)
                    name_to_profile_index[canonical_key] = profile_index
                    profile_names.append(canonical_key)
                    selected_rows.append(local_row)
                else:
                    duplicate_names += 1

                name_to_profile_index.setdefault(raw_key, profile_index)

            if selected_rows:
                np.save(tmp_profiles, profile_matrix[selected_rows].copy())

            rows_seen += len(chunk)
            print(
                f"   Profile rows scanned: {rows_seen:,}; "
                f"unique profiles: {len(profile_names):,}"
            )

    if not profile_names:
        raise ValueError("No transcriptomic profiles were loaded.")

    compact_appended_arrays(
        paths.tmp_profile_blocks,
        paths.profiles,
        shape=(len(profile_names), len(gene_columns)),
        dtype=np.float32,
    )
    paths.tmp_profile_blocks.unlink(missing_ok=True)

    write_lines(paths.profile_names, profile_names)
    write_lines(paths.gene_names, gene_columns)

    alias_df = pd.DataFrame(
        {
            "alias": list(name_to_profile_index.keys()),
            "profile_index": list(name_to_profile_index.values()),
        }
    ).sort_values(["profile_index", "alias"])
    alias_df.to_csv(paths.alias_map, index=False)

    print(
        f"Saved {len(profile_names):,} drug profiles x {len(gene_columns):,} genes "
        f"to {paths.profiles}"
    )
    if duplicate_names:
        print(f"   Skipped {duplicate_names:,} duplicate transcriptomic drug names.")

    return name_to_profile_index, profile_names, gene_columns


def build_patient_profile_indices(
    patient_file: Path,
    patient_drug_column: str,
    name_map: dict[str, str],
    name_to_profile_index: dict[str, int],
    paths: CompactPaths,
    *,
    max_drugs: int,
    chunk_size: int,
    max_rows: int | None,
    excluded_terms: tuple[str, ...],
) -> dict[str, object]:
    print(f"Building FAERS patient -> transcriptomic profile index map from {patient_file}...")

    rows_processed = 0
    rows_with_match = 0
    rows_skipped_too_many_drugs = 0
    drug_slots_seen = 0
    matched_drug_slots = 0
    unmatched_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()

    read_csv_kwargs = {
        "usecols": [patient_drug_column],
        "dtype": {patient_drug_column: str},
        "chunksize": chunk_size,
        "low_memory": False,
    }
    if max_rows is not None:
        read_csv_kwargs["nrows"] = max_rows

    with paths.tmp_patient_blocks.open("ab") as tmp_patient_indices:
        for chunk in pd.read_csv(patient_file, **read_csv_kwargs):
            chunk_indices = np.full((len(chunk), max_drugs), -1, dtype=np.int32)

            for local_row, drug_string in enumerate(chunk[patient_drug_column].values):
                row_has_match = False
                drugs, excluded_drugs = parse_drug_list(
                    drug_string,
                    max_drugs,
                    excluded_terms,
                )
                excluded_counter.update(excluded_drugs)
                if len(drugs) > max_drugs:
                    rows_skipped_too_many_drugs += 1
                    continue

                drug_slots_seen += len(drugs)

                for slot_index, raw_drug in enumerate(drugs):
                    resolved_drug = resolve_drug_name(raw_drug, name_map)
                    profile_index = name_to_profile_index.get(resolved_drug)

                    if profile_index is None:
                        profile_index = name_to_profile_index.get(raw_drug)

                    if profile_index is None:
                        unmatched_counter[resolved_drug] += 1
                        continue

                    chunk_indices[local_row, slot_index] = profile_index
                    matched_drug_slots += 1
                    row_has_match = True

                if row_has_match:
                    rows_with_match += 1

            np.save(tmp_patient_indices, chunk_indices)
            rows_processed += len(chunk)
            print(
                f"   Patient rows processed: {rows_processed:,}; "
                f"rows with transcriptomic match: {rows_with_match:,}"
            )

    compact_appended_arrays(
        paths.tmp_patient_blocks,
        paths.patient_indices,
        shape=(rows_processed, max_drugs),
        dtype=np.int32,
    )
    paths.tmp_patient_blocks.unlink(missing_ok=True)

    print(f"Saved patient profile indices to {paths.patient_indices}")

    return {
        "patient_rows": rows_processed,
        "rows_with_transcriptomic_match": rows_with_match,
        "rows_skipped_too_many_drugs": rows_skipped_too_many_drugs,
        "drug_slots_seen": drug_slots_seen,
        "matched_drug_slots": matched_drug_slots,
        "excluded_drug_slots": sum(excluded_counter.values()),
        "row_match_rate": rows_with_match / rows_processed if rows_processed else 0.0,
        "drug_slot_match_rate": matched_drug_slots / drug_slots_seen
        if drug_slots_seen
        else 0.0,
        "top_unmatched_drugs": unmatched_counter.most_common(25),
        "top_excluded_drugs": excluded_counter.most_common(25),
    }


def write_stats(paths: CompactPaths, stats: dict[str, object]) -> None:
    with paths.stats.open("w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")


def build_compact_artifacts(
    *,
    patient_file: Path,
    transcriptomic_file: Path,
    mapping_file: Path,
    output_dir: Path,
    prefix: str,
    patient_drug_column: str | None,
    transcriptomic_drug_column: str | None,
    max_drugs: int,
    patient_chunk_size: int,
    transcriptomic_chunk_size: int,
    max_patients: int | None,
    max_transcriptomic_rows: int | None,
    csr_output_file: Path | None,
    excluded_terms: tuple[str, ...],
) -> tuple[CompactPaths, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = CompactPaths(output_dir=output_dir, prefix=prefix)
    remove_stale_outputs(paths, csr_output_file)

    transcriptomic_sep = csv_separator(transcriptomic_file)
    detected_transcriptomic_drug_column = transcriptomic_drug_column or detect_column(
        transcriptomic_file,
        preferred_names=["drug_name", "perturbation", "compound", "perturbagen"],
        keyword_fallbacks=["drug", "compound", "perturb"],
        sep=transcriptomic_sep,
        allow_first_column=True,
    )
    detected_patient_drug_column = patient_drug_column or detect_column(
        patient_file,
        preferred_names=["drug_combination", "Drug_Combination"],
        keyword_fallbacks=["drug", "combo"],
    )

    print(f"Using transcriptomic drug column: {detected_transcriptomic_drug_column}")
    print(f"Using FAERS drug column: {detected_patient_drug_column}")
    print(f"Loading drug-name mapping from {mapping_file}...")
    name_map = load_name_map(mapping_file)
    print(f"Loaded {len(name_map):,} normalized drug-name mappings.")
    print(f"Using {len(excluded_terms):,} non-drug exclusion terms.")

    name_to_profile_index, profile_names, gene_columns = load_transcriptomic_profiles(
        transcriptomic_file,
        detected_transcriptomic_drug_column,
        name_map,
        paths,
        chunk_size=transcriptomic_chunk_size,
        max_rows=max_transcriptomic_rows,
    )
    patient_stats = build_patient_profile_indices(
        patient_file,
        detected_patient_drug_column,
        name_map,
        name_to_profile_index,
        paths,
        max_drugs=max_drugs,
        chunk_size=patient_chunk_size,
        max_rows=max_patients,
        excluded_terms=excluded_terms,
    )

    stats: dict[str, object] = {
        "patient_file": str(patient_file),
        "transcriptomic_file": str(transcriptomic_file),
        "mapping_file": str(mapping_file),
        "patient_drug_column": detected_patient_drug_column,
        "transcriptomic_drug_column": detected_transcriptomic_drug_column,
        "max_drugs": max_drugs,
        "excluded_drug_terms": list(excluded_terms),
        "gene_count": len(gene_columns),
        "profile_count": len(profile_names),
        "profiles_file": str(paths.profiles),
        "patient_indices_file": str(paths.patient_indices),
        "profile_names_file": str(paths.profile_names),
        "gene_names_file": str(paths.gene_names),
        "alias_map_file": str(paths.alias_map),
        **patient_stats,
    }
    write_stats(paths, stats)

    print(f"Saved build stats to {paths.stats}")
    return paths, stats


def patient_batch_to_dense_profiles(
    profile_matrix: np.ndarray,
    patient_indices: np.ndarray,
) -> np.ndarray:
    batch = np.zeros((len(patient_indices), profile_matrix.shape[1]), dtype=np.float32)
    valid_counts = np.zeros(len(patient_indices), dtype=np.float32)

    for slot_index in range(patient_indices.shape[1]):
        profile_indices = patient_indices[:, slot_index]
        valid_mask = profile_indices >= 0

        if not valid_mask.any():
            continue

        batch[valid_mask] += profile_matrix[profile_indices[valid_mask]]
        valid_counts[valid_mask] += 1

    matched_mask = valid_counts > 0
    batch[matched_mask] /= valid_counts[matched_mask, None]

    return batch


def materialize_patient_csr(
    paths: CompactPaths,
    csr_output_file: Path,
    *,
    chunk_size: int,
    allow_huge: bool,
    stats: dict[str, object],
) -> dict[str, object]:
    profile_matrix = np.load(paths.profiles, mmap_mode="r")
    patient_indices = np.load(paths.patient_indices, mmap_mode="r")

    patient_rows = patient_indices.shape[0]
    gene_count = profile_matrix.shape[1]
    matched_rows = int(stats["rows_with_transcriptomic_match"])
    estimated_nonzeros = matched_rows * gene_count

    if estimated_nonzeros > HUGE_MATERIALIZED_NNZ_LIMIT and not allow_huge:
        raise RuntimeError(
            "Refusing to materialize a huge patient x gene CSR matrix. "
            f"Estimated nonzeros: {estimated_nonzeros:,}. "
            "Use the compact artifacts for training, add --max-patients for a small "
            "debug matrix, or pass --allow-huge-materialization if you really want it."
        )

    print(
        f"Materializing CSR matrix to {csr_output_file} "
        f"({patient_rows:,} patients x {gene_count:,} genes)..."
    )

    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    nonzeros = 0

    for start in range(0, patient_rows, chunk_size):
        end = min(start + chunk_size, patient_rows)
        dense_batch = patient_batch_to_dense_profiles(
            profile_matrix,
            patient_indices[start:end],
        )

        local_rows, local_cols = np.nonzero(dense_batch)
        values = dense_batch[local_rows, local_cols].astype(np.float32, copy=False)

        row_blocks.append((local_rows + start).astype(np.int32, copy=False))
        col_blocks.append(local_cols.astype(np.int32, copy=False))
        value_blocks.append(values)
        nonzeros += len(values)

        print(f"   CSR rows materialized: {end:,}; nonzeros: {nonzeros:,}")

    if row_blocks:
        rows = np.concatenate(row_blocks)
        cols = np.concatenate(col_blocks)
        values = np.concatenate(value_blocks)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float32)

    csr_matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(patient_rows, gene_count),
        dtype=np.float32,
    )
    sparse.save_npz(csr_output_file, csr_matrix)

    csr_stats = {
        "csr_output_file": str(csr_output_file),
        "csr_shape": list(csr_matrix.shape),
        "csr_nonzeros": int(csr_matrix.nnz),
        "csr_density": float(
            csr_matrix.nnz / (csr_matrix.shape[0] * csr_matrix.shape[1])
        )
        if csr_matrix.shape[0] and csr_matrix.shape[1]
        else 0.0,
    }

    print(
        f"Saved CSR matrix with shape {csr_matrix.shape} and "
        f"{csr_matrix.nnz:,} nonzeros."
    )
    return csr_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build transcriptomic FAERS artifacts aligned to drug cocktails. "
            "The default compact format avoids duplicating dense gene vectors for "
            "millions of patients."
        )
    )
    parser.add_argument("--patient-file", type=Path, default=DEFAULT_PATIENT_FILE)
    parser.add_argument(
        "--transcriptomic-file",
        type=Path,
        default=DEFAULT_TRANSCRIPTOMIC_FILE,
    )
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="transcriptomic")
    parser.add_argument("--patient-drug-column")
    parser.add_argument("--transcriptomic-drug-column")
    parser.add_argument("--max-drugs", type=int, default=DEFAULT_MAX_DRUGS)
    parser.add_argument(
        "--patient-chunk-size",
        type=int,
        default=DEFAULT_PATIENT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--transcriptomic-chunk-size",
        type=int,
        default=DEFAULT_TRANSCRIPTOMIC_CHUNK_SIZE,
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Limit FAERS rows for a quick debug build.",
    )
    parser.add_argument(
        "--max-transcriptomic-rows",
        type=int,
        help="Limit perturb rows for a quick debug build.",
    )
    parser.add_argument(
        "--exclude-drug-term",
        action="append",
        default=[],
        help=(
            "Additional normalized substring to skip in FAERS drug cocktails. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Disable the built-in cosmetic/non-drug exclusion terms.",
    )
    parser.add_argument(
        "--mode",
        choices=["compact", "csr", "both"],
        default="compact",
        help="Use compact for training-scale data; csr is intended for small debug runs.",
    )
    parser.add_argument(
        "--csr-output-file",
        type=Path,
        help="Where to save the optional materialized patient x gene CSR matrix.",
    )
    parser.add_argument(
        "--materialize-chunk-size",
        type=int,
        default=512,
        help="Patient rows per dense batch when materializing CSR.",
    )
    parser.add_argument(
        "--allow-huge-materialization",
        action="store_true",
        help="Override the guard that blocks enormous patient x gene CSR outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csr_output_file = args.csr_output_file
    if csr_output_file is None and args.mode in {"csr", "both"}:
        csr_output_file = args.output_dir / f"{args.prefix}_patient_gene_sparse.npz"
    excluded_terms = build_excluded_terms(
        args.exclude_drug_term,
        include_defaults=not args.no_default_exclusions,
    )

    paths, stats = build_compact_artifacts(
        patient_file=args.patient_file,
        transcriptomic_file=args.transcriptomic_file,
        mapping_file=args.mapping_file,
        output_dir=args.output_dir,
        prefix=args.prefix,
        patient_drug_column=args.patient_drug_column,
        transcriptomic_drug_column=args.transcriptomic_drug_column,
        max_drugs=args.max_drugs,
        patient_chunk_size=args.patient_chunk_size,
        transcriptomic_chunk_size=args.transcriptomic_chunk_size,
        max_patients=args.max_patients,
        max_transcriptomic_rows=args.max_transcriptomic_rows,
        csr_output_file=csr_output_file,
        excluded_terms=excluded_terms,
    )

    if args.mode in {"csr", "both"}:
        assert csr_output_file is not None
        csr_stats = materialize_patient_csr(
            paths,
            csr_output_file,
            chunk_size=args.materialize_chunk_size,
            allow_huge=args.allow_huge_materialization,
            stats=stats,
        )
        stats["materialized_csr"] = csr_stats
        write_stats(paths, stats)

    print("\nDone.")
    print("Compact artifacts:")
    print(f"  profiles:        {paths.profiles}")
    print(f"  patient indices: {paths.patient_indices}")
    print(f"  genes:           {paths.gene_names}")
    print(f"  stats:           {paths.stats}")


if __name__ == "__main__":
    main()
