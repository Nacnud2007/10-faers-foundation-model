"""Validate the multimodal autoencoder against FAERS drug combinations."""
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
import re

from autoencoder import (
    MultimodalAutoencoder,
    load_sparse_npz_fast,
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_TRANS_PROFILES_PATH,
    DEFAULT_TRANS_PATIENT_INDICES_PATH,
)

CHECKPOINT = Path("/Users/duncanpark/10-faers-foundation-model/output/autoencoder/multimodal/multimodal_autoencoder_epoch13.pt")
COMBO_FILE = Path("/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv")

MIN_REPORTS = 5              # min FAERS reports for a drug to be considered at all
MIN_TRANS_MATCHED_ROWS = 5   # min rows with a real transcriptomic match to trust the "real" input
N_DRUGS_TO_CHECK = 50        # how many single drugs to sample for this check

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

X = load_sparse_npz_fast(DEFAULT_X_PATH)  # chemical
Y = load_sparse_npz_fast(DEFAULT_Y_PATH)  # clinical
trans_profiles = np.load(DEFAULT_TRANS_PROFILES_PATH, mmap_mode="r")
trans_patient_indices = np.load(DEFAULT_TRANS_PATIENT_INDICES_PATH, mmap_mode="r")

TRANS_DIM = trans_profiles.shape[1]

model = MultimodalAutoencoder(
    trans_dim=TRANS_DIM,
    clinical_dim=Y.shape[1],
    dropout=checkpoint["args"]["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def clean_drug_string(text):
    if pd.isna(text):
        return ()
    raw_drugs = re.split(r'[,;]', str(text).lower())
    cleaned = [d.strip().rstrip('.') for d in raw_drugs if d.strip()]
    return tuple(sorted(set(cleaned)))


df = pd.read_csv(COMBO_FILE)
df["drugs"] = df["drug_combination"].apply(clean_drug_string)
tuple_to_indices = df.groupby("drugs").indices

single_drugs = [k for k in tuple_to_indices.keys() if len(k) == 1]
print(f"Found {len(single_drugs)} single drugs total.")


def get_real_trans_mean(idx):
    """Average real trans profile across rows in idx that have at least one
    matched transcriptomic slot -- mirrors BatchedTrimodalLoader's aggregation
    during training, so 'real' here means what the model actually trained on."""
    profile_index_batch = trans_patient_indices[idx]  # (n_rows, max_drugs)
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
    n_matched = int(matched_mask.sum())
    if n_matched < MIN_TRANS_MATCHED_ROWS:
        return None, n_matched
    return trans_rows[matched_mask].mean(axis=0, keepdims=True), n_matched


print("Comparing real-vs-zero trans_in for drugs with real transcriptomic coverage...")
checked = 0
rows = []
real_trans_outputs = {}  # drug -> trans_recon under real trans_in, for the reference-scale calc below

for drug in single_drugs:
    if checked >= N_DRUGS_TO_CHECK:
        break

    idx = tuple_to_indices[drug]
    if len(idx) < MIN_REPORTS:
        continue

    real_trans_mean, n_matched = get_real_trans_mean(idx)
    if real_trans_mean is None:
        continue

    chem_mean = np.asarray(X[idx].mean(axis=0))
    clin_mean = np.asarray(Y[idx].mean(axis=0))

    chem_in = torch.FloatTensor(chem_mean).to(device)
    clin_in = torch.FloatTensor(clin_mean).to(device)
    real_trans_in = torch.FloatTensor(real_trans_mean).to(device)
    zero_trans_in = torch.zeros_like(real_trans_in).to(device)

    with torch.no_grad():
        trans_recon_real, _, _ = model(chem_in=chem_in, trans_in=real_trans_in, clin_in=clin_in)
        trans_recon_zero, _, _ = model(chem_in=chem_in, trans_in=zero_trans_in, clin_in=clin_in)

    real_out = trans_recon_real.cpu().numpy().reshape(1, -1)
    zero_out = trans_recon_zero.cpu().numpy().reshape(1, -1)

    dist = cosine_distances(real_out, zero_out)[0][0]

    rows.append({
        "drug": drug[0],
        "n_reports": len(idx),
        "n_trans_matched": n_matched,
        "real_vs_zero_distance": dist,
    })
    real_trans_outputs[drug[0]] = real_out
    checked += 1

results = pd.DataFrame(rows)
print(f"\nChecked {len(results)} drugs with real transcriptomic coverage.\n")
print(results.sort_values("real_vs_zero_distance", ascending=False))
print("\nreal_vs_zero_distance summary:")
print(results["real_vs_zero_distance"].describe())

# Reference scale: how far apart are two DIFFERENT drugs' real-trans-conditioned
# outputs from each other, typically? If real_vs_zero_distance is small relative
# to this, zero-filling is a safe stand-in. If it's comparable or larger, the
# zero-fill output is dominated by the missing input, not a real signal.
drug_names = list(real_trans_outputs.keys())
if len(drug_names) >= 2:
    all_real_outputs = np.vstack([real_trans_outputs[d] for d in drug_names])
    between_drug_distances = cosine_distances(all_real_outputs)
    upper_tri = between_drug_distances[np.triu_indices_from(between_drug_distances, k=1)]
    print("\nFor reference -- distance between DIFFERENT drugs' real-trans outputs:")
    print(pd.Series(upper_tri).describe())

results.to_csv("trans_zero_fill_validation.csv", index=False)
