#!/usr/bin/env python3
"""
align_lincs_to_faers.py

Maps LINCS L1000 signatures directly to FAERS PubChem CIDs and standardized names.
Preserves individual signatures without destructive averaging.
"""

from __future__ import annotations

import gzip
import re
import time
from pathlib import Path
import pandas as pd
import pubchempy as pcp

PROJECT_ROOT = Path(__file__).resolve().parents[0]
RAW_GMT = PROJECT_ROOT / "data" / "raw" / "lincs-l1000-cp.gmt.gz"
FAERS_MAPPING_CSV = PROJECT_ROOT / "data" / "drug_name_mapping_updated.csv"
OUTPUT_MAPPED_CSV = PROJECT_ROOT / "pseudobulk_lincs_faers_mapped.csv"


def clean_lincs_drug_name(name: str) -> str:
    """Aligns with your FAERS cleaning rules."""
    name = str(name).strip().rstrip('.')
    name = re.sub(r'/\d+/', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    return ' '.join(name.split()).lower()


def load_faers_dictionary(mapping_csv_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    """Loads FAERS mapping CSV into lookup dicts for string -> CID and string -> standard_name."""
    print(f"1. Loading FAERS drug mapping from {mapping_csv_path}...")
    df = pd.read_csv(mapping_csv_path)
    
    # Map both messy_name and standardized_name to CID
    name_to_cid = {}
    name_to_standard = {}
    
    for _, row in df.iterrows():
        cid = row["pubchem_cid"] if pd.notna(row["pubchem_cid"]) else None
        std_name = str(row["standardized_name"]).lower() if pd.notna(row["standardized_name"]) else ""
        
        if pd.notna(row["messy_name"]):
            messy = str(row["messy_name"]).lower().strip()
            if cid:
                name_to_cid[messy] = int(cid)
            name_to_standard[messy] = std_name
            
        if std_name:
            if cid:
                name_to_cid[std_name] = int(cid)
            name_to_standard[std_name] = std_name
            
    print(f"✓ Loaded {len(name_to_cid):,} PubChem CID mappings from FAERS dictionary.")
    return name_to_cid, name_to_standard


def parse_and_map_lincs(gmt_gz_path: Path, name_to_cid: dict, name_to_standard: dict) -> pd.DataFrame:
    print("\n2. Parsing LINCS GMT and mapping to FAERS CIDs...")
    
    records = []
    all_genes = set()
    unmapped_drugs = set()
    
    with gzip.open(gmt_gz_path, "rt", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            
            term = parts[0]
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if not genes:
                continue
            
            # Extract raw drug term from LINCS signature header
            raw_drug = term.split("::")[0].split("_")[0].strip()
            clean_drug = clean_lincs_drug_name(raw_drug)
            
            # Lookup in FAERS dictionary
            cid = name_to_cid.get(clean_drug, None)
            standard_name = name_to_standard.get(clean_drug, clean_drug)
            
            if cid is None:
                unmapped_drugs.add(clean_drug)
                
            records.append({
                "signature_id": term,
                "lincs_raw_drug": raw_drug,
                "pubchem_cid": cid,
                "standardized_name": standard_name,
                "gene_set": set(genes)
            })
            all_genes.update(genes)

    print(f"✓ Processed {len(records):,} total signatures across {len(all_genes):,} genes.")
    print(f"   Mapped signatures: {sum(1 for r in records if r['pubchem_cid'] is not None):,}")
    print(f"   Unmapped LINCS unique compounds: {len(unmapped_drugs):,}")
    
    # Construct final dense matrix
    print("\n3. Building final mapped feature matrix...")
    sorted_genes = sorted(all_genes)
    output_rows = []
    
    for rec in records:
        row = {
            "signature_id": rec["signature_id"],
            "pubchem_cid": rec["pubchem_cid"],
            "standardized_name": rec["standardized_name"]
        }
        g_set = rec["gene_set"]
        for g in sorted_genes:
            row[g] = 1.0 if g in g_set else 0.0
            
        output_rows.append(row)

    df = pd.DataFrame(output_rows)
    return df


def main():
    if not FAERS_MAPPING_CSV.exists():
        raise FileNotFoundError(f"Missing FAERS dictionary at {FAERS_MAPPING_CSV}. Run your mapping scripts first.")
        
    name_to_cid, name_to_standard = load_faers_dictionary(FAERS_MAPPING_CSV)
    df_mapped = parse_and_map_lincs(RAW_GMT, name_to_cid, name_to_standard)
    
    print(f"\n4. Saving mapped dataset to {OUTPUT_MAPPED_CSV}...")
    df_mapped.to_csv(OUTPUT_MAPPED_CSV, index=False)
    
    print("\nDone! Dataset Summary:")
    print(f"  Total Signatures (Rows): {df_mapped.shape[0]:,}")
    print(f"  Total Columns: {df_mapped.shape[1]:,} (3 metadata + {df_mapped.shape[1]-3:,} gene features)")
    print(f"  Signatures with Valid FAERS PubChem CID: {df_mapped['pubchem_cid'].notna().sum():,}")


if __name__ == "__main__":
    main()