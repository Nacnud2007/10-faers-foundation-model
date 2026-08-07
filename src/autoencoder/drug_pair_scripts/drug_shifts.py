"""Analyze drug shift patterns using the trained autoencoder embeddings."""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from pathlib import Path
import re
import time
from datetime import datetime, timedelta

from autoencoder import load_sparse_npz_fast, DEFAULT_Y_PATH

COMBO_FILE = Path("/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv")

MIN_REPORTS = 5
PROGRESS_EVERY = 1000  # print progress every 1000 pairs

# Load data
start_time = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ADR matrix...")

Y = load_sparse_npz_fast(DEFAULT_Y_PATH)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading FAERS combinations...")

df = pd.read_csv(COMBO_FILE)


def clean_drug_string(text):
    if pd.isna(text):
        return ()
    raw_drugs = re.split(r'[,;]', str(text).lower())
    cleaned = [d.strip().rstrip('.') for d in raw_drugs if d.strip()]
    return tuple(sorted(set(cleaned)))


df["drugs"] = df["drug_combination"].apply(clean_drug_string)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Mapping drug row positions...")

tuple_to_indices = df.groupby("drugs").indices

drug_pairs = [k for k in tuple_to_indices.keys() if len(k) == 2]

print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(drug_pairs):,} drug pairs.")


def get_adr_profile(drug_tuple):
    idx = tuple_to_indices.get(drug_tuple)

    if idx is None or len(idx) < MIN_REPORTS:
        return None

    return np.asarray(Y[idx].mean(axis=0)).reshape(1, -1)


single_cache = {}

def get_cached_single(drug):
    key = (drug,)
    if key not in single_cache:
        single_cache[key] = get_adr_profile(key)
    return single_cache[key]


# Main loop
print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing ADR profile shifts...\n")

results = []

loop_start = time.time()
last_print = loop_start

for i, (drug_a, drug_b) in enumerate(drug_pairs, start=1):

    combo_profile = get_adr_profile((drug_a, drug_b))
    a_profile = get_cached_single(drug_a)
    b_profile = get_cached_single(drug_b)

    if combo_profile is None or a_profile is None or b_profile is None:
        continue

    shift_a = cosine_distances(combo_profile, a_profile)[0][0]
    shift_b = cosine_distances(combo_profile, b_profile)[0][0]
    individual_difference = cosine_distances(a_profile, b_profile)[0][0]

    results.append(
        {
            "drug_A": drug_a,
            "drug_B": drug_b,
            "n_combo_reports": len(tuple_to_indices[(drug_a, drug_b)]),
            "n_A_reports": len(tuple_to_indices.get((drug_a,), [])),
            "n_B_reports": len(tuple_to_indices.get((drug_b,), [])),
            "combo_vs_A": shift_a,
            "combo_vs_B": shift_b,
            "A_vs_B": individual_difference,
            "interaction_score": (shift_a + shift_b) / 2,
        }
    )

    # Progress reporting
    if i % PROGRESS_EVERY == 0 or i == len(drug_pairs):

        elapsed = time.time() - loop_start
        pairs_per_sec = i / elapsed

        remaining = len(drug_pairs) - i
        eta_seconds = remaining / pairs_per_sec if pairs_per_sec > 0 else 0

        finish = datetime.now() + timedelta(seconds=eta_seconds)

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"{i:,}/{len(drug_pairs):,} "
            f"({100*i/len(drug_pairs):5.1f}%) | "
            f"{pairs_per_sec:6.1f} pairs/sec | "
            f"Elapsed {timedelta(seconds=int(elapsed))} | "
            f"ETA {timedelta(seconds=int(eta_seconds))} | "
            f"Finish ~ {finish.strftime('%H:%M:%S')}"
        )


# Save
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving results...")

results = pd.DataFrame(results)

results = results.sort_values(
    ["interaction_score", "n_combo_reports"],
    ascending=[False, False]
)

results.to_csv("top_ddi_adr_shifts.csv", index=False)

total_elapsed = time.time() - start_time

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Finished.")
print(f"Total runtime: {timedelta(seconds=int(total_elapsed))}")

print("\nTop 20:")
print(results.head(20))

print("\nInteraction score summary:")
print(results["interaction_score"].describe())

print(
    "\nPairs at ceiling:",
    (results["interaction_score"] == 1.0).sum(),
    "of",
    len(results)
)

ceiling = results[results["interaction_score"] == 1.0]

print(
    ceiling[
        [
            "drug_A",
            "drug_B",
            "n_combo_reports",
            "n_A_reports",
            "n_B_reports",
        ]
    ].sort_values("n_combo_reports")
)
