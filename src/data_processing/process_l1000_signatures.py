#!/usr/bin/env python3
"""
Process L1000-CP Pruned CD Coefficients (.h5ad) using the local GMT file
to bridge plate-well annotations into a collapsed [Drugs x Genes] pseudobulk matrix.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp

# --- Configuration Paths ---
MATRIX_FILE_NAME = "l1000-cp-pruned-cd-coefficients.h5ad"
GMT_FILE_NAME = "lincs-l1000-cp.gmt"

CURRENT_DIR = Path(__file__).resolve().parent          # src/autoencoder
PROJECT_ROOT = CURRENT_DIR.parent.parent              # 10-faers-foundation-model
DATA_DIR = PROJECT_ROOT / "data"

SIGNATURE_MATRIX_PATH = DATA_DIR / MATRIX_FILE_NAME
GMT_PATH = DATA_DIR / GMT_FILE_NAME
OUTPUT_FILE_PATH = PROJECT_ROOT / "pseudobulk_perturb.csv"

NAME_CLEAN_RE = re.compile(r"[^a-z0-9]+")

def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned = NAME_CLEAN_RE.sub(" ", str(value).strip().lower())
    return " ".join(cleaned.split())

def parse_gmt_to_bridge(gmt_path: Path) -> dict[str, str]:
    """
    Directly extracts the matrix key tokens out of the GMT string header.
    """
    print(f"Parsing local signature definitions from {gmt_path.name}...")
    sig_to_drug = {}
    
    # Precise regex tokens to pull plate and well coordinates directly
    plate_pattern = re.compile(r'(REP\.[A-Z0-9]+|MOAR[0-9]+)')
    well_pattern = re.compile(r':([A-Z][0-9]{2})\b')
    
    with open(gmt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
                
            header = line.split("\t")[0].strip()
            parts = header.split("::")
            if len(parts) < 2:
                continue
                
            drug_name = parts[0].strip()
            cell_line = parts[1].strip()
            
            # Determine time signature contextually
            time_point = "6H" if "::6H::" in header or "6H" in parts else "24H"
            
            plate_match = plate_pattern.search(header)
            well_match = well_pattern.search(header)
            
            if plate_match and well_match:
                plate_id = plate_match.group(1)
                well_id = well_match.group(1)
                
                # Reconstruct key format to exactly match the matrix index: 'REP.B007_HT29_24H:C23'
                matrix_key = f"{plate_id}_{cell_line}_{time_point}:{well_id}"
                sig_to_drug[matrix_key] = normalize_name(drug_name)
                
    return sig_to_drug

def main():
    if not SIGNATURE_MATRIX_PATH.exists() or not GMT_PATH.exists():
        raise FileNotFoundError(f"Verify files exist in: {DATA_DIR}")
    
    # 1. Build the mapping dictionary
    sig_to_drug_map = parse_gmt_to_bridge(GMT_PATH)
    print(f"   Successfully parsed {len(sig_to_drug_map):,} structural mappings from GMT definitions.")

    # 2. Load the Expression Matrix from H5AD
    print(f"Reading signature matrix from {MATRIX_FILE_NAME}...")
    adata = sc.read_h5ad(SIGNATURE_MATRIX_PATH)
    
    X_data = adata.X
    if sp.issparse(X_data):
        X_data = X_data.toarray()
        
    df = pd.DataFrame(X_data, index=adata.obs_names, columns=adata.var_names)
    print(f"   Loaded data matrix shape: {df.shape}")

    # 3. Apply the mapping key
    print("Mapping matrix index rows to unique drug targets...")
    clean_index_names = [sig_to_drug_map.get(str(idx).strip(), "") for idx in df.index]
    df.index = clean_index_names
    df.index.name = "drug_name"
    
    # Drop rows that couldn't be resolved
    initial_count = len(df)
    df = df[df.index != ""]
    print(f"   Retained {len(df):,}/{initial_count:,} signatures matching valid drug IDs.")

    if len(df) == 0:
        print("\n[CRITICAL ERROR]: Retained rows dropped to 0.")
        print("Review key structures:")
        print(f"Matrix index samples: {list(adata.obs_names[:3])}")
        print(f"Parsed mapping sample keys: {list(sig_to_drug_map.keys())[:3]}")
        return

    # 4. Collapse duplicates across multi-condition signatures
    print(f"   Collapsing multi-condition signatures into unique drug averages...")
    collapsed_df = df.groupby(level=0, sort=True).mean()
    
    # 5. Format and save to root directory
    collapsed_df = collapsed_df.reset_index()
    
    print(f"Saving finalized pseudobulk matrix to {OUTPUT_FILE_PATH}...")
    collapsed_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Processing Complete! Output Shape: {collapsed_df.shape} (Unique Drugs: {len(collapsed_df):,})")

if __name__ == "__main__":
    main()