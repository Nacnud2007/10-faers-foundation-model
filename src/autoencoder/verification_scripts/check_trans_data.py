"""
Standalone diagnostic -- does NOT touch the running training process.
Checks whether trans_profiles/patient_indices could explain trans loss staying at 0.0000.
"""
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # same convention as autoencoder.py

trans_profiles = np.load(PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_drug_profiles.npy", mmap_mode="r")
patient_indices = np.load(PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_patient_profile_indices.npy", mmap_mode="r")

print(f"trans_profiles shape: {trans_profiles.shape}")
print(f"patient_indices shape: {patient_indices.shape}")

# How many patients have ZERO matched drug slots (all -1)?
sample = patient_indices[:200_000]  # sample for speed, not the full 14.8M
unmatched_rows = (sample < 0).all(axis=1).sum()
print(f"\nOf {len(sample):,} sampled patients: {unmatched_rows:,} "
      f"({unmatched_rows/len(sample)*100:.1f}%) have NO matched transcriptomic profile "
      "(all 5 slots are -1, so their trans_row defaults to all-zeros).")

# What do the actual profile values look like in scale?
profile_sample = np.asarray(trans_profiles[:5000])
print(f"\ntrans_profiles value stats (sample of 5000 profiles):")
print(f"  mean={profile_sample.mean():.6f}  std={profile_sample.std():.6f}")
print(f"  min={profile_sample.min():.6f}  max={profile_sample.max():.6f}")
print(f"  fraction exactly zero: {(profile_sample == 0).mean()*100:.2f}%")