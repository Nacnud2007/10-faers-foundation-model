"""Analyze transcriptomic prediction shifts with the multimodal autoencoder."""
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
import re

from autoencoder import (
    MultimodalAutoencoder,
    load_sparse_npz_fast,
    DEFAULT_X_PATH,   # chemical
    DEFAULT_Y_PATH,   # clinical
)

CHECKPOINT = Path("/Users/duncanpark/10-faers-foundation-model/output/autoencoder/multimodal/multimodal_autoencoder_epoch13.pt")
COMBO_FILE = Path("/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv")

MIN_REPORTS = 5
TRANS_DIM = 12327 # same as checkpoint

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

X = load_sparse_npz_fast(DEFAULT_X_PATH)  # chemical
Y = load_sparse_npz_fast(DEFAULT_Y_PATH)  # clinical

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
drug_pairs = [k for k in tuple_to_indices.keys() if len(k) == 2]
print(f"Found {len(drug_pairs)} drug pairs to analyze.")


def get_predicted_trans(drug_tuple):
    idx = tuple_to_indices.get(drug_tuple)
    if idx is None or len(idx) < MIN_REPORTS:
        return None

    chem_mean = np.asarray(X[idx].mean(axis=0))
    clin_mean = np.asarray(Y[idx].mean(axis=0))

    chem_in = torch.FloatTensor(chem_mean).to(device)
    clin_in = torch.FloatTensor(clin_mean).to(device)
    trans_in = torch.zeros((chem_in.shape[0], TRANS_DIM)).to(device)

    with torch.no_grad():
        trans_recon, clin_recon, z = model(chem_in=chem_in, trans_in=trans_in, clin_in=clin_in)

    return trans_recon.cpu().numpy().reshape(1, -1)


single_cache = {}
def get_cached_single_trans(drug):
    key = (drug,)
    if key not in single_cache:
        single_cache[key] = get_predicted_trans(key)
    return single_cache[key]


print("Computing predicted transcriptomic shifts...")
results = []
for drug_a, drug_b in drug_pairs:
    combo_trans = get_predicted_trans((drug_a, drug_b))
    a_trans = get_cached_single_trans(drug_a)
    b_trans = get_cached_single_trans(drug_b)

    if combo_trans is None or a_trans is None or b_trans is None:
        continue

    shift_a = cosine_distances(combo_trans, a_trans)[0][0]
    shift_b = cosine_distances(combo_trans, b_trans)[0][0]

    results.append(
        {
            "drug_A": drug_a,
            "drug_B": drug_b,
            "trans_combo_vs_A": shift_a,
            "trans_combo_vs_B": shift_b,
            "trans_interaction_score": (shift_a + shift_b) / 2,
        }
    )

results = pd.DataFrame(results)
results.to_csv("trans_predicted_shifts.csv", index=False)
print(results.sort_values("trans_interaction_score", ascending=False).head(20))
