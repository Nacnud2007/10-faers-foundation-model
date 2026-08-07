"""Inspect a multimodal autoencoder checkpoint and print its stored losses."""
import torch
from pathlib import Path
from pprint import pprint

checkpoint_path = Path("/Users/duncanpark/10-faers-foundation-model/output/autoencoder/multimodal/multimodal_autoencoder_epoch03.pt")

ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("Epoch:", ckpt["epoch"])

print("\nLosses:")
print(ckpt["losses"])
