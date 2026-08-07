"""Probe the transcript encoder for similarity shifts across FAERS cases."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import from your actual model file
from autoencoder import (
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
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
Y = load_sparse_npz_fast(DEFAULT_Y_PATH)
trans_profiles = np.load(DEFAULT_TRANS_PROFILES_PATH, mmap_mode="r")
trans_patient_indices = np.load(DEFAULT_TRANS_PATIENT_INDICES_PATH, mmap_mode="r")
TRANS_DIM = trans_profiles.shape[1]

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only = False)

# Instantiate the correct MultimodalADRPredictor model
model = MultimodalADRPredictor(
    trans_dim=TRANS_DIM,
    clinical_dim=Y.shape[1],
    dropout=checkpoint["args"].get("dropout", 0.05),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
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


def get_real_trans_mean(idx):
    profile_index_batch = trans_patient_indices[idx]
    trans_rows = np.zeros((len(idx), TRANS_DIM), dtype=np.float32)
    valid_counts = np.zeros(len(idx), dtype=np.float32)
    for slot in range(profile_index_batch.shape[1]):
        slot_indices = profile_index_batch[:, slot]
        valid_mask = slot_indices >= 0
        if not valid_mask.any():
            continue
        trans_rows[valid_mask] += trans_profiles[slot_indices[valid_mask]]
        valid_counts[valid_mask] += 1
    matched_mask = valid_counts > 0
    if matched_mask.sum() < 5:
        return None
    return trans_rows[matched_mask].mean(axis=0, keepdims=True)


sample_trans = []
for drug in single_drugs:
    idx = tuple_to_indices[drug]
    if len(idx) < 20:
        continue
    t = get_real_trans_mean(idx)
    if t is None:
        continue
    sample_trans.append((drug[0], t))
    if len(sample_trans) >= 5:
        break

print(
    "Checking model.trans_encoder output directly (before fusion) for real drug profiles + zero:\n"
)

zero_input = np.zeros((1, TRANS_DIM), dtype=np.float32)
sample_trans.append(("ZERO_INPUT", zero_input))

with torch.no_grad():
    for name, trans_arr in sample_trans:
        trans_in = torch.from_numpy(trans_arr).to(device)
        trans_emb = model.trans_encoder(trans_in)
        n_nonzero = (trans_emb.abs() > 1e-8).sum().item()
        print(
            f"{name:20s} | trans_emb nonzero units: {n_nonzero}/{trans_emb.shape[1]} "
            f"| min={trans_emb.min().item():.6f} max={trans_emb.max().item():.6f}"
        )
