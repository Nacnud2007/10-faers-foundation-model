"""Run an expanded validation pass over the multimodal autoencoder outputs."""
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

MIN_REPORTS_FOR_ANCHOR = 20
MIN_TRANS_MATCHED_ROWS = 5
N_DONOR_DRUGS = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

X = load_sparse_npz_fast(DEFAULT_X_PATH)
Y = load_sparse_npz_fast(DEFAULT_Y_PATH)
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
    if matched_mask.sum() < MIN_TRANS_MATCHED_ROWS:
        return None
    return trans_rows[matched_mask].mean(axis=0, keepdims=True)


# Pick one "anchor" drug -- chem_in/clin_in will stay fixed to this drug's
# values for the whole test. Only trans_in changes.
anchor_drug = None
anchor_idx = None
anchor_trans = None
donor_trans_profiles = {}

for drug in single_drugs:
    idx = tuple_to_indices[drug]
    if len(idx) < MIN_REPORTS_FOR_ANCHOR:
        continue
    real_trans = get_real_trans_mean(idx)
    if real_trans is None:
        continue
    if anchor_drug is None:
        anchor_drug = drug
        anchor_idx = idx
        anchor_trans = real_trans
        continue
    donor_trans_profiles[drug[0]] = real_trans
    if len(donor_trans_profiles) >= N_DONOR_DRUGS:
        break

print(f"Anchor drug (chem_in/clin_in held fixed to this): {anchor_drug[0]}")
print(f"Donor drugs (only trans_in swapped in): {list(donor_trans_profiles.keys())}\n")

chem_mean = np.asarray(X[anchor_idx].mean(axis=0))
clin_mean = np.asarray(Y[anchor_idx].mean(axis=0))
chem_in = torch.FloatTensor(chem_mean).to(device)
clin_in = torch.FloatTensor(clin_mean).to(device)


def run_with_trans(trans_arr):
    trans_in = torch.FloatTensor(trans_arr).to(device)
    with torch.no_grad():
        trans_recon, _, _ = model(chem_in=chem_in, trans_in=trans_in, clin_in=clin_in)
    return trans_recon.cpu().numpy().reshape(1, -1)


own_output = run_with_trans(anchor_trans)
zero_output = run_with_trans(np.zeros_like(anchor_trans))
own_vs_zero = cosine_distances(own_output, zero_output)[0][0]
print(f"anchor's own real trans_in vs zero trans_in: {own_vs_zero:.6e}\n")

print("Swapping in OTHER drugs' real trans_in (chem_in/clin_in unchanged):")
rows = []
for donor_name, donor_trans in donor_trans_profiles.items():
    donor_output = run_with_trans(donor_trans)
    dist = cosine_distances(own_output, donor_output)[0][0]
    rows.append({
        "donor_drug": donor_name,
        "trans_recon_distance_from_anchor_output": dist,
        "donor_trans_mean_abs_value": float(np.abs(donor_trans).mean()),
    })

results = pd.DataFrame(rows).sort_values("trans_recon_distance_from_anchor_output", ascending=False)
print(results.to_string(index=False))
print(f"\nFor reference, anchor's own trans mean abs value: {float(np.abs(anchor_trans).mean()):.6e}")
print(
    "\nIf these distances are all still ~1e-6 or smaller (same order as the "
    "real-vs-zero check), trans_in is not meaningfully affecting trans_recon -- "
    "the model isn't using the transcriptomic branch for this output. If they're "
    "comparable to the between-drug reference distances from the previous check "
    "(median ~0.17), trans_in does matter and the near-zero real-vs-zero result "
    "was just because real trans values happen to be tiny in magnitude."
)
