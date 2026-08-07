"""Compare multimodal autoencoder checkpoint losses across epochs."""
import os
import torch
import pandas as pd

CHECKPOINT_DIR = "/Users/duncanpark/10-faers-foundation-model/output/autoencoder/multimodal"

rows = []

for file in sorted(os.listdir(CHECKPOINT_DIR)):
    if not file.startswith("multimodal_autoencoder_epoch"):
        continue

    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, file),
        map_location="cpu",
        weights_only=False,
    )

    losses = checkpoint["losses"]

    rows.append({
        "epoch": checkpoint["epoch"],
        "total": losses["total"],
        "trans": losses["trans"],
        "clin": losses["clin"],
        "predicted_positive_rate": losses["predicted_positive_rate"],
        "true_positive_rate": losses["true_positive_rate"],
    })

df = pd.DataFrame(rows).sort_values("epoch")

print(df)

print("\nChanges from previous epoch:\n")
print(df.set_index("epoch").diff())

df.to_csv("epoch_metrics.csv", index=False)
