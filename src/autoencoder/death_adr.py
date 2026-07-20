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
default_adr_vocab = project_root / "adr_vocabulary.txt"

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
    def __init__(self, *, slot_bits: int = fingerprint_bits, embed_dim: int = default_slot_embed_dim, dropout: float = 0.0) -> None:
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
        self.slot_embedder = SharedSlotEmbedder(slot_bits=slot_bits, embed_dim=slot_embed_dim, dropout = dropout)
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
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(self.activation(self.hidden(latent))))


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
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = DrugEncoder(
            slot_embed_dim=slot_embed_dim,
            hidden_dim=chemical_hidden_dim,
            latent_dim=latent_dim,
            dropout = dropout
        )
        self.decoder = AdverseEventDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            dropout = dropout
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



def choose_row_indices(total_rows: int, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or max_rows >= total_rows:
        return np.arange(total_rows, dtype=np.int64)

    if max_rows < 2:
        raise ValueError("max_rows must be at least 2 so the train/validation split works.")

    random_generator = np.random.default_rng(seed)
    row_indices = random_generator.choice(total_rows, size=max_rows, replace=False)
    row_indices.sort()
    return row_indices.astype(np.int64, copy=False)


def find_label_positive_indices(adverse_event_matrix: sparse.csr_matrix, column_index: int) -> np.ndarray:
    '''
    Returns every row index where the given adverse-event column is nonzero, i.e. every
    row where the target ADR (Death) was actually reported.
    '''
    column = adverse_event_matrix[:, [column_index]].tocsc()
    positive_rows = np.unique(column.nonzero()[0])
    return positive_rows.astype(np.int64, copy=False)


def build_stratified_row_indices(
    total_rows: int,
    positive_row_indices: np.ndarray,
    *,
    negative_per_positive: float,
    max_rows: int | None,
    seed: int,
) -> np.ndarray:
    '''
    Keeps every positive (Death-reporting) row and fills in a matching pool of negative
    rows at the requested ratio. Plain random sampling over 14.8M rows only captures a
    small, proportional slice of the ~666k positives (e.g. ~180k out of 666k at
    max_rows=4,000,000); this keeps them all so the rare positive class isn't thrown away.
    If max_rows still forces a cut, positives are kept in full and only negatives are thinned.
    '''
    random_generator = np.random.default_rng(seed)

    positive_row_indices = np.unique(positive_row_indices)
    negative_mask = np.ones(total_rows, dtype=bool)
    negative_mask[positive_row_indices] = False
    negative_pool = np.flatnonzero(negative_mask)

    n_negatives_wanted = int(round(len(positive_row_indices) * negative_per_positive))
    n_negatives = min(n_negatives_wanted, len(negative_pool))
    negative_row_indices = random_generator.choice(negative_pool, size=n_negatives, replace=False)

    if max_rows is not None and max_rows < len(positive_row_indices) + len(negative_row_indices):
        n_negatives_capped = max(0, max_rows - len(positive_row_indices))
        if n_negatives_capped < len(negative_row_indices):
            negative_row_indices = random_generator.choice(
                negative_row_indices, size=n_negatives_capped, replace=False
            )

    combined = np.concatenate([positive_row_indices, negative_row_indices])
    combined.sort()
    return combined.astype(np.int64, copy=False)


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
    parser.add_argument("--max-rows", type=int, default=None,
        help="Optional hard cap on total training rows. Positives are always kept in full; "
             "only negatives get thinned if this cap forces a cut. Default: no cap "
             "(use every positive plus --neg-per-pos-ratio negatives).")
    parser.add_argument("--neg-per-pos-ratio", type=float, default=4.0,
        help="Number of negative (non-Death) rows to sample per positive (Death) row. "
             "With ~666k Death-positive rows, a ratio of 4.0 yields ~3.3M training rows "
             "with a ~20%% positive rate, instead of the <5%% positive rate you'd get "
             "from a blind random sample of the full 14.8M rows.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-max", type=float, default=3.0,
        help="Upper clamp on pos_weight passed to BCEWithLogitsLoss. Lower = fewer false positives, higher = more recall.")
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slot-embed-dim", type=int, default=default_slot_embed_dim)
    parser.add_argument("--chemical-hidden-dim", type=int, default=default_chemical_hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=default_latent_dim)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64,
        help="Decoder hidden width. The general multi-ADR script defaults to 1024 because it "
             "outputs thousands of ADR logits; here output_dim=1 (Death only), so a 256->64->1 "
             "decoder is plenty and trains faster with less overfitting risk.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a single forward pass and exit.")
    parser.add_argument(
        "--adr-vocab",
        type=Path,
        default=default_adr_vocab,
        help="adr_vocabulary.txt generated when building Y_train_sparse.npz.",
    )   
    
    return parser.parse_args()


def evaluate_predictions(model, loader, device, threshold=0.5):
    """
    Evaluates the model on validation data and returns Precision, Recall, and F1-Score.
    """
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for chemical_batch, adverse_event_batch in loader:
            chemical_batch = chemical_batch.to(device)
            logits, _ = model(chemical_batch)
            
            # Convert logits to probabilities, then to binary predictions (0 or 1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            all_targets.append(adverse_event_batch.cpu())
            all_preds.append(preds.cpu())
            
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1


def collect_val_predictions(
    model: DrugToAdverseEventAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the model once over the validation loader and returns raw probabilities
    and targets, so the threshold sweep and per-class breakdown below don't each
    need their own forward pass over the data.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for chemical_batch, adverse_event_batch in loader:
            chemical_batch = chemical_batch.to(device)
            logits, _ = model(chemical_batch)
            probs = torch.sigmoid(logits)

            all_targets.append(adverse_event_batch.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()
    return y_true, y_probs


def sweep_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """
    Computes precision/recall/F1 at each threshold in `thresholds` and returns
    the full sweep plus the single entry with the highest F1. A fixed 0.5 cutoff
    is rarely optimal when most of the 2,048 ADR columns are rare.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    results: list[dict] = []
    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(np.float32)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    best = max(results, key=lambda entry: entry["f1"])
    return results, best


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

    # Look up the specific single vocabulary index for Death
    adr_vocab = args.adr_vocab.read_text().splitlines()
    adr_to_index = {adr.strip(): i for i, adr in enumerate(adr_vocab)}

    # Adjust "Death" string if your vocabulary is capitalized differently (e.g., "DEATH")
    target_adr = "Death" 
    if target_adr not in adr_to_index:
        raise ValueError(f"'{target_adr}' not found in vocabulary.")

    selected_columns = np.array([adr_to_index[target_adr]], dtype=np.int64)
    selected_counts = np.asarray(adverse_event_matrix[:, selected_columns].sum(axis=0)).ravel().astype(np.int64)    

    print(f"Selected adverse-event outputs for training: {len(selected_columns):,}")
    print(f"Total '{target_adr}' occurrences in full dataset: {int(selected_counts[0]):,}")

    # Find every row in the FULL dataset with the selected ADR, BEFORE subsampling —
    # this ADR is rare (~4.5% of rows), so a blind random sample would only capture a
    # small, proportional slice of the positives. Instead keep every positive row and
    # sample negatives at a controlled ratio.
    target_column_index = int(selected_columns[0])
    positive_row_indices = find_label_positive_indices(adverse_event_matrix, target_column_index)
    print(f"Rows where '{target_adr}' was reported: {len(positive_row_indices):,}")

    selected_row_indices = build_stratified_row_indices(
        chemical_matrix.shape[0],
        positive_row_indices,
        negative_per_positive=args.neg_per_pos_ratio,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    n_positive_selected = len(positive_row_indices)
    n_negative_selected = len(selected_row_indices) - n_positive_selected
    print(
        f"Stratified training set: {len(selected_row_indices):,} rows "
        f"({n_positive_selected:,} positive / {n_negative_selected:,} negative, "
        f"positive rate={100 * n_positive_selected / len(selected_row_indices):.1f}%)"
    )

    chemical_subset = chemical_matrix[selected_row_indices]
    adverse_event_subset = (
        adverse_event_matrix[selected_row_indices][:, selected_columns]
    ).tocsr()

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
    pos_weight = torch.clamp(negative_counts / positive_counts, min=1.0, max=args.pos_weight_max)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = DrugToAdverseEventAutoencoder(
        output_dim=len(selected_columns),
        slot_embed_dim=args.slot_embed_dim,
        chemical_hidden_dim=args.chemical_hidden_dim,
        latent_dim=args.latent_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout = args.dropout
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
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        val_prec, val_rec, val_f1 = evaluate_predictions(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"Precision={val_prec:.4f} | Recall={val_rec:.4f} | F1-Score={val_f1:.4f}"
        )

        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            print(
                f"Stopping early: val_loss has not improved for "
                f"{epochs_since_improvement} epochs (patience={args.early_stopping_patience})."
            )
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "death_adr_encoder.pt"
    metadata_path = args.output_dir / "death_adr_encoder.json"
    latent_path = args.output_dir / "death_adr_latents.npz"
    selected_columns_path = args.output_dir / "death_adr_selected_adr_columns.npy"
    selected_counts_path = args.output_dir / "death_selected_adr_counts.npy"

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\nRunning final evaluation on best checkpoint...")
    val_y_true, val_y_probs = collect_val_predictions(model, val_loader, device)

    threshold_sweep, best_threshold_entry = sweep_thresholds(val_y_true, val_y_probs)
    print(
        f"Best threshold={best_threshold_entry['threshold']:.2f} | "
        f"Precision={best_threshold_entry['precision']:.4f} | "
        f"Recall={best_threshold_entry['recall']:.4f} | "
        f"F1={best_threshold_entry['f1']:.4f}"
    )

    torch.save(
        {
            "model_state_dict": best_state_dict,
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
        "positive_row_count": int(n_positive_selected),
        "negative_row_count": int(n_negative_selected),
        "neg_per_pos_ratio_requested": args.neg_per_pos_ratio,
        "positive_rate_in_training_set": n_positive_selected / len(selected_row_indices),
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
        "final_evaluation": {
            "threshold_sweep": threshold_sweep,
            "best_threshold": best_threshold_entry,
            "frequency_buckets": bucket_results,
        },
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