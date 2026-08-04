"""
Trains a chemical autoencoder that reconstructs drug fingerprints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, Subset


project_root = Path(__file__).resolve().parents[2]

default_x_path = project_root / "X_train_sparse.npz"
default_output_directory = project_root / "output" / "autoencoder" / "chemical_autoencoder"

max_drugs = 5
fingerprint_bits = 3_095
chemical_input_dim = max_drugs * fingerprint_bits

default_slot_embed_dim = 256
default_chemical_hidden_dim = 1024
default_latent_dim = 512
default_decoder_hidden_dim = 1024


class SharedSlotEmbedder(nn.Module):
    '''
    Takes 3,095-bit drug slot and turns it into a dense vector; the same weights
    are reused for all 5 drug slots.
    '''
    def __init__(self, *, slot_bits: int = fingerprint_bits, embed_dim: int = default_slot_embed_dim, dropout: float = 0.00) -> None:
        super().__init__()
        self.linear = nn.Linear(slot_bits, embed_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, slot_tensor: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(slot_tensor)))


class DrugEncoder(nn.Module):
    '''
    Reshapes the full drug cocktail into five slots, embeds each slot, flattens it, and compresses
    everything back together into the latent vector
    '''
    def __init__(self, *, max_drugs: int = max_drugs,
        slot_bits: int = fingerprint_bits,
        slot_embed_dim: int = default_slot_embed_dim,
        hidden_dim: int = default_chemical_hidden_dim,
        latent_dim: int = default_latent_dim,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_drugs = max_drugs
        self.slot_bits = slot_bits
        self.slot_embedder = SharedSlotEmbedder(slot_bits=slot_bits, embed_dim=slot_embed_dim, dropout=dropout)
        self.chemical_hidden = nn.Linear(max_drugs * slot_embed_dim, hidden_dim)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, chemical_input: torch.Tensor) -> torch.Tensor:
        if chemical_input.dim() == 2:
            chemical_input = chemical_input.reshape(
                chemical_input.size(0), self.max_drugs, self.slot_bits
            )

        slot_embeddings = self.slot_embedder(chemical_input)
        flattened_slots = slot_embeddings.reshape(slot_embeddings.size(0), -1)
        chemical_hidden = self.dropout(self.activation(self.chemical_hidden(flattened_slots)))
        return self.activation(self.latent_layer(chemical_hidden))


class ChemicalDecoder(nn.Module):
    '''
    Expands the latent vector back into the original 3,095-bit chemical fingerprint logits
    '''
    def __init__(
        self,
        *,
        latent_dim: int = default_latent_dim,
        hidden_dim: int = default_decoder_hidden_dim,
        output_dim: int = chemical_input_dim,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(self.activation(self.hidden(latent))))


class ChemicalAutoencoder(nn.Module):
    '''
    Wraps the chemical encoder and decoder to perform self-reconstruction of fingerprints.
    '''
    def __init__(
        self,
        *,
        output_dim: int = chemical_input_dim,
        slot_embed_dim: int = default_slot_embed_dim,
        chemical_hidden_dim: int = default_chemical_hidden_dim,
        latent_dim: int = default_latent_dim,
        decoder_hidden_dim: int = default_decoder_hidden_dim,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = DrugEncoder(
            slot_embed_dim=slot_embed_dim,
            hidden_dim=chemical_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.decoder = ChemicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(self, chemical_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(chemical_input)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class SparseChemicalDataset(Dataset):
    '''
    Loads chemical rows from X_train_sparse.npz and returns them as both input and target.
    '''
    def __init__(self, chemical_matrix: sparse.csr_matrix) -> None:
        self.chemical_matrix = chemical_matrix.tocsr()

    def __len__(self) -> int:
        return self.chemical_matrix.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chemical_row = self.chemical_matrix[index].toarray().astype(np.float32, copy=False).ravel()
        chemical_tensor = torch.from_numpy(chemical_row)
        return chemical_tensor, chemical_tensor


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def choose_row_indices(total_rows: int, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or max_rows >= total_rows:
        return np.arange(total_rows, dtype=np.int64)

    if max_rows < 2:
        raise ValueError("max_rows must be at least 2 so the train/validation split works.")

    random_generator = np.random.default_rng(seed)
    row_indices = random_generator.choice(total_rows, size=max_rows, replace=False)
    row_indices.sort()
    return row_indices.astype(np.int64, copy=False)


def split_indices(total_rows: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if total_rows < 2:
        raise ValueError("Need at least two rows to split into train and validation sets.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    random_generator = np.random.default_rng(seed)
    permutation = random_generator.permutation(total_rows)

    validation_rows = max(1, int(round(total_rows * validation_fraction)))
    if validation_rows >= total_rows:
        validation_rows = total_rows - 1

    validation_indices = permutation[:validation_rows]
    training_indices = permutation[validation_rows:]
    return training_indices.astype(np.int64, copy=False), validation_indices.astype(np.int64, copy=False)


def build_dataloaders(
    chemical_matrix: sparse.csr_matrix,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, SparseChemicalDataset]:
    full_dataset = SparseChemicalDataset(chemical_matrix)
    training_indices, validation_indices = split_indices(len(full_dataset), validation_fraction, seed)

    training_subset = Subset(full_dataset, training_indices.tolist())
    validation_subset = Subset(full_dataset, validation_indices.tolist())

    train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, full_dataset


def run_epoch(
    model: ChemicalAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    for chemical_batch, target_batch in loader:
        chemical_batch = chemical_batch.to(device)
        target_batch = target_batch.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        reconstruction_logits, _ = model(chemical_batch)
        loss = criterion(reconstruction_logits, target_batch)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = chemical_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def evaluate_reconstruction(model: ChemicalAutoencoder, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    """
    Evaluates how accurately the autoencoder reconstructs the binary fingerprint bits.
    """
    model.eval()
    total_correct = 0
    total_bits = 0

    with torch.no_grad():
        for chemical_batch, target_batch in loader:
            chemical_batch = chemical_batch.to(device)
            target_batch = target_batch.to(device)

            logits, _ = model(chemical_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            total_correct += (preds == target_batch).sum().item()
            total_bits += target_batch.numel()

    bit_accuracy = total_correct / max(total_bits, 1)
    return bit_accuracy


def export_latent_embeddings(
    model: ChemicalAutoencoder,
    dataset: SparseChemicalDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for chemical_batch, _ in loader:
            chemical_batch = chemical_batch.to(device)
            _, latent = model(chemical_batch)
            latent_batches.append(latent.cpu().numpy())

    return np.concatenate(latent_batches, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a chemical autoencoder to reconstruct fingerprint bits from FAERS sparse matrices."
    )
    parser.add_argument("--x-path", type=Path, default=default_x_path)
    parser.add_argument("--output-dir", type=Path, default=default_output_directory)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slot-embed-dim", type=int, default=default_slot_embed_dim)
    parser.add_argument("--chemical-hidden-dim", type=int, default=default_chemical_hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=default_latent_dim)
    parser.add_argument("--decoder-hidden-dim", type=int, default=default_decoder_hidden_dim)
    parser.add_argument("--smoke-test", action="store_true", help="Run a single forward pass and exit.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("FAERS Chemical Autoencoder (Step 1)")
    print("=" * 72)

    chemical_matrix = sparse.load_npz(args.x_path).tocsr()

    if chemical_matrix.shape[1] != chemical_input_dim:
        raise ValueError(
            f"Expected X to have {chemical_input_dim:,} columns, got {chemical_matrix.shape[1]:,}."
        )

    print(f"Rows in full dataset: {chemical_matrix.shape[0]:,}")
    print(f"Chemical input width: {chemical_matrix.shape[1]:,}")

    selected_row_indices = choose_row_indices(chemical_matrix.shape[0], args.max_rows, args.seed)
    print(f"Selected rows for training: {len(selected_row_indices):,}")

    chemical_subset = chemical_matrix[selected_row_indices]

    train_loader, val_loader, full_dataset = build_dataloaders(
        chemical_subset,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = ChemicalAutoencoder(
        output_dim=chemical_input_dim,
        slot_embed_dim=args.slot_embed_dim,
        chemical_hidden_dim=args.chemical_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(f"Model parameters: {count_parameters(model):,}")

    if args.smoke_test:
        sample_chemical, sample_target = next(iter(train_loader))
        sample_chemical = sample_chemical.to(device)
        sample_target = sample_target.to(device)
        with torch.no_grad():
            reconstruction, latent = model(sample_chemical)
            loss = criterion(reconstruction, sample_target)
        print("Smoke test passed.")
        print(f"  chemical batch:       {tuple(sample_chemical.shape)}")
        print(f"  latent batch:         {tuple(latent.shape)}")
        print(f"  reconstruction batch: {tuple(reconstruction.shape)}")
        print(f"  loss:                 {loss.item():.6f}")
        return

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        val_acc = evaluate_reconstruction(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"Bit Accuracy={val_acc:.4%}"
        )

        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            print(
                f"Stopping early: val_loss has not improved for "
                f"{epochs_since_improvement} epochs (patience={args.early_stopping_patience})."
            )
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "chemical_autoencoder.pt"
    metadata_path = args.output_dir / "chemical_autoencoder.json"
    latent_path = args.output_dir / "chemical_latents.npz"

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "model_config": {
                "slot_embed_dim": args.slot_embed_dim,
                "chemical_hidden_dim": args.chemical_hidden_dim,
                "latent_dim": args.latent_dim,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "output_dim": chemical_input_dim,
            },
        },
        checkpoint_path,
    )

    latent_embeddings = export_latent_embeddings(
        model, full_dataset, device=device, batch_size=args.batch_size
    )
    np.savez_compressed(
        latent_path,
        latent_embeddings=latent_embeddings,
        selected_row_indices=selected_row_indices,
    )

    metadata = {
        "x_path": str(args.x_path),
        "output_dir": str(args.output_dir),
        "raw_chemical_shape": list(chemical_matrix.shape),
        "selected_row_count": int(len(selected_row_indices)),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "latent_path": str(latent_path),
        "best_val_loss": best_val_loss,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nSaved artifacts:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  metadata:   {metadata_path}")
    print(f"  latents:    {latent_path}")


if __name__ == "__main__":
    main()