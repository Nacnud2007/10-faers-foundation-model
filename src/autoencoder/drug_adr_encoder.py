"""
Trains a chemical encoder that maps drug cocktails to ADR outputs.
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
default_y_path = project_root / "Y_train_sparse.npz"
default_output_directory = project_root / "output" / "autoencoder" / "drug_adr_encoder"

max_drugs = 5
fingerprint_bits = 3_095
chemical_input_dim = max_drugs * fingerprint_bits

default_slot_embed_dim = 128
default_chemical_hidden_dim = 256
default_latent_dim = 128
default_decoder_hidden_dim = 512


class SharedSlotEmbedder(nn.Module):
    '''
    Takes 3,095-bit drug slot and turns it into a dense vector; the same weights
    are reused for all 5 drug slots.
    '''
    def __init__(self, *, slot_bits: int = fingerprint_bits, embed_dim: int = default_slot_embed_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(slot_bits, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, slot_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(slot_tensor))


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
    ) -> None:
        super().__init__()
        self.max_drugs = max_drugs
        self.slot_bits = slot_bits
        self.slot_embedder = SharedSlotEmbedder(slot_bits=slot_bits, embed_dim=slot_embed_dim)
        self.chemical_hidden = nn.Linear(max_drugs * slot_embed_dim, hidden_dim)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, chemical_input: torch.Tensor) -> torch.Tensor:
        if chemical_input.dim() == 2:
            chemical_input = chemical_input.reshape(
                chemical_input.size(0), self.max_drugs, self.slot_bits
            )

        slot_embeddings = self.slot_embedder(chemical_input)
        flattened_slots = slot_embeddings.reshape(slot_embeddings.size(0), -1)
        chemical_hidden = self.activation(self.chemical_hidden(flattened_slots))
        return self.activation(self.latent_layer(chemical_hidden))


class AdverseEventDecoder(nn.Module):
    '''
    Expands the latent vector back into logits for the chosen ADR outputs
    '''
    def __init__(
        self,
        *,
        latent_dim: int = default_latent_dim,
        hidden_dim: int = default_decoder_hidden_dim,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(latent)))


class DrugToAdverseEventAutoencoder(nn.Module):
    '''
    Wraps the encoder and decoder and its forward pass returns the tuple
    (ADR_logits, latent_vector).
    '''
    def __init__(
        self,
        *,
        output_dim: int,
        slot_embed_dim: int = default_slot_embed_dim,
        chemical_hidden_dim: int = default_chemical_hidden_dim,
        latent_dim: int = default_latent_dim,
        decoder_hidden_dim: int = default_decoder_hidden_dim,
    ) -> None:
        super().__init__()
        self.encoder = DrugEncoder(
            slot_embed_dim=slot_embed_dim,
            hidden_dim=chemical_hidden_dim,
            latent_dim=latent_dim,
        )
        self.decoder = AdverseEventDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, chemical_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(chemical_input)
        logits = self.decoder(latent)
        return logits, latent


class SparseDrugEventDataset(Dataset):
    '''
    Loads one row from X_train_sparse.npz and Y_train_sparse.npz and returns dense tensors for training
    '''
    def __init__(self, chemical_matrix: sparse.csr_matrix, adverse_event_matrix: sparse.csr_matrix) -> None:
        self.chemical_matrix = chemical_matrix.tocsr()
        self.adverse_event_matrix = adverse_event_matrix.tocsr()

    def __len__(self) -> int:
        return self.chemical_matrix.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chemical_row = self.chemical_matrix[index].toarray().astype(np.float32, copy=False).ravel()
        adverse_event_row = self.adverse_event_matrix[index].toarray().astype(np.float32, copy=False).ravel()
        return torch.from_numpy(chemical_row), torch.from_numpy(adverse_event_row)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def choose_top_adverse_event_columns(adverse_event_matrix: sparse.csr_matrix, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k_adrs must be greater than zero.")

    label_counts = np.asarray(adverse_event_matrix.sum(axis=0)).ravel().astype(np.int64, copy=False)
    nonzero_columns = np.flatnonzero(label_counts)

    if nonzero_columns.size == 0:
        raise ValueError("No active adverse-event columns were found in Y_train_sparse.npz.")

    if top_k >= nonzero_columns.size:
        selected_columns = nonzero_columns[np.argsort(label_counts[nonzero_columns])[::-1]]
    else:
        candidate_counts = label_counts[nonzero_columns]
        top_positions = np.argpartition(candidate_counts, -top_k)[-top_k:]
        selected_columns = nonzero_columns[top_positions]
        selected_columns = selected_columns[np.argsort(label_counts[selected_columns])[::-1]]

    selected_counts = label_counts[selected_columns]
    return selected_columns.astype(np.int64, copy=False), selected_counts.astype(np.int64, copy=False)


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
    adverse_event_matrix: sparse.csr_matrix,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, SparseDrugEventDataset]:
    full_dataset = SparseDrugEventDataset(chemical_matrix, adverse_event_matrix)
    training_indices, validation_indices = split_indices(len(full_dataset), validation_fraction, seed)

    training_subset = Subset(full_dataset, training_indices.tolist())
    validation_subset = Subset(full_dataset, validation_indices.tolist())

    train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, full_dataset


def run_epoch(
    model: DrugToAdverseEventAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0

    for chemical_batch, adverse_event_batch in loader:
        chemical_batch = chemical_batch.to(device)
        adverse_event_batch = adverse_event_batch.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits, _ = model(chemical_batch)
        loss = criterion(logits, adverse_event_batch)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = chemical_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def export_latent_embeddings(
    model: DrugToAdverseEventAutoencoder,
    dataset: SparseDrugEventDataset,
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
        description="Train a chemical encoder and adverse-event decoder from the FAERS sparse matrices."
    )
    parser.add_argument("--x-path", type=Path, default=default_x_path)
    parser.add_argument("--y-path", type=Path, default=default_y_path)
    parser.add_argument("--output-dir", type=Path, default=default_output_directory)
    parser.add_argument("--top-k-adrs", type=int, default=2_048)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slot-embed-dim", type=int, default=default_slot_embed_dim)
    parser.add_argument("--chemical-hidden-dim", type=int, default=default_chemical_hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=default_latent_dim)
    parser.add_argument("--decoder-hidden-dim", type=int, default=default_decoder_hidden_dim)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("FAERS Chemical Encoder / Adverse-Event Decoder")
    print("=" * 72)

    chemical_matrix = sparse.load_npz(args.x_path).tocsr()
    adverse_event_matrix = sparse.load_npz(args.y_path).tocsr()

    if chemical_matrix.shape[0] != adverse_event_matrix.shape[0]:
        raise ValueError(
            "Chemical and adverse-event row counts do not match: "
            f"{chemical_matrix.shape[0]:,} vs {adverse_event_matrix.shape[0]:,}."
        )

    if chemical_matrix.shape[1] != chemical_input_dim:
        raise ValueError(
            f"Expected X to have {chemical_input_dim:,} columns, got {chemical_matrix.shape[1]:,}."
        )

    print(f"Rows in full dataset: {chemical_matrix.shape[0]:,}")
    print(f"Chemical input width: {chemical_matrix.shape[1]:,}")
    print(f"Raw adverse-event width: {adverse_event_matrix.shape[1]:,}")

    selected_columns, selected_counts = choose_top_adverse_event_columns(
        adverse_event_matrix, args.top_k_adrs
    )
    print(f"Selected adverse-event outputs: {len(selected_columns):,}")

    selected_row_indices = choose_row_indices(
        chemical_matrix.shape[0], args.max_rows, args.seed
    )
    print(f"Selected rows for training: {len(selected_row_indices):,}")

    chemical_subset = chemical_matrix[selected_row_indices]
    adverse_event_subset = adverse_event_matrix[selected_row_indices][:, selected_columns].tocsr()

    train_loader, val_loader, full_dataset = build_dataloaders(
        chemical_subset,
        adverse_event_subset,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    adverse_event_counts = np.asarray(adverse_event_subset.sum(axis=0)).ravel().astype(np.float32, copy=False)
    positive_counts = torch.from_numpy(np.maximum(adverse_event_counts, 1.0))
    negative_counts = float(len(full_dataset)) - positive_counts
    pos_weight = torch.clamp(negative_counts / positive_counts, min=1.0, max=1_000.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DrugToAdverseEventAutoencoder(
        output_dim=len(selected_columns),
        slot_embed_dim=args.slot_embed_dim,
        chemical_hidden_dim=args.chemical_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(f"Model parameters: {count_parameters(model):,}")

    if args.smoke_test:
        sample_chemical, sample_target = next(iter(train_loader))
        sample_chemical = sample_chemical.to(device)
        sample_target = sample_target.to(device)
        with torch.no_grad():
            logits, latent = model(sample_chemical)
            loss = criterion(logits, sample_target)
        print("Smoke test passed.")
        print(f"  chemical batch: {tuple(sample_chemical.shape)}")
        print(f"  latent batch:   {tuple(latent.shape)}")
        print(f"  output batch:   {tuple(logits.shape)}")
        print(f"  loss:           {loss.item():.6f}")
        return

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "drug_adr_encoder.pt"
    metadata_path = args.output_dir / "drug_adr_encoder.json"
    latent_path = args.output_dir / "drug_adr_latents.npz"
    selected_columns_path = args.output_dir / "selected_adr_columns.npy"
    selected_counts_path = args.output_dir / "selected_adr_counts.npy"

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "slot_embed_dim": args.slot_embed_dim,
                "chemical_hidden_dim": args.chemical_hidden_dim,
                "latent_dim": args.latent_dim,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "output_dim": len(selected_columns),
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
        selected_adr_columns=selected_columns,
    )
    np.save(selected_columns_path, selected_columns)
    np.save(selected_counts_path, selected_counts)

    metadata = {
        "x_path": str(args.x_path),
        "y_path": str(args.y_path),
        "output_dir": str(args.output_dir),
        "raw_chemical_shape": list(chemical_matrix.shape),
        "raw_adverse_event_shape": list(adverse_event_matrix.shape),
        "selected_row_count": int(len(selected_row_indices)),
        "selected_adr_count": int(len(selected_columns)),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "latent_path": str(latent_path),
        "selected_columns_path": str(selected_columns_path),
        "selected_counts_path": str(selected_counts_path),
        "best_val_loss": best_val_loss,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nSaved artifacts:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  metadata:   {metadata_path}")
    print(f"  latents:    {latent_path}")
    print(f"  columns:    {selected_columns_path}")
    print(f"  counts:     {selected_counts_path}")


if __name__ == "__main__":
    main()
