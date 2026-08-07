"""Probe the chemical encoder for similarity shifts across FAERS cases."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import from your actual model file
from autoencoder import (
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    MultimodalADRPredictor,
    load_sparse_npz_fast,
)

CHECKPOINT = Path(
    "/Users/duncanpark/10-faers-foundation-model/output/models/adr_predictor/latest.pt"
)
COMBO_FILE = Path(
    "/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv"
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load data and state
X = load_sparse_npz_fast(DEFAULT_X_PATH)
Y = load_sparse_npz_fast(DEFAULT_Y_PATH)
CHEM_DIM = X.shape[1]

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

# Instantiate the MultimodalADRPredictor model
model = MultimodalADRPredictor(
    trans_dim=0,    # Set to 0 since transcriptomic data is not here
    clinical_dim=Y.shape[1],
    dropout=checkpoint["args"].get("dropout", 0.05),
).to(device)

# Filter out trans_encoder weights
state_dict = checkpoint["model_state_dict"]
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("trans_encoder")}
model.load_state_dict(filtered_state_dict, strict=False)

model.eval()

def clean_drug_string(text):
    if pd.isna(text):
        return ()
    raw_drugs = re.split(r"[,;]", str(text).lower())
    cleaned = [d.strip().rstrip(".") for d in raw_drugs if d.strip()]
    return tuple(sorted(set(cleaned)))


df = pd.read_csv(COMBO_FILE)
df["drugs"] = df["drug_combination"].apply(clean_drug_string)
tuple_to_indices = df.groupby("drugs").indices
single_drugs = [k for k in tuple_to_indices.keys() if len(k) == 1]


def get_real_chem_mean(idx):
    chem_rows = X[idx].toarray().astype(np.float32)
    if chem_rows.shape[0] < 5:
        return None
    return chem_rows.mean(axis=0, keepdims=True)


sample_chem = []
for drug in single_drugs:
    idx = tuple_to_indices[drug]
    if len(idx) < 20:
        continue
    c = get_real_chem_mean(idx)
    if c is None:
        continue
    sample_chem.append((drug[0], c))
    if len(sample_chem) >= 5:
        break

print(
    "Checking model.chem_encoder output directly (before fusion) for real drug profiles + zero:\n"
)

zero_input = np.zeros((1, CHEM_DIM), dtype=np.float32)
sample_chem.append(("ZERO_INPUT", zero_input))

with torch.no_grad():
    for name, chem_arr in sample_chem:
        chem_in = torch.from_numpy(chem_arr).to(device)
        chem_emb = model.chem_encoder(chem_in)
        n_nonzero = (chem_emb.abs() > 1e-8).sum().item()
        print(
            f"{name:20s} | chem_emb nonzero units: {n_nonzero}/{chem_emb.shape[1]} "
            f"| min={chem_emb.min().item():.6f} max={chem_emb.max().item():.6f}"
        )
