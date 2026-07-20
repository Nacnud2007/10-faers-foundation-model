"""
chemical + transcriptomic + clinical → shared bottleneck

Reconstruct transcriptomic and clinical modalities from a shared latent space.
The chemical encoder is used only to provide information to the latent representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

# Reuse chemical encoder
from drug_adr_encoder import DrugEncoder, default_slot_embed_dim, default_chemical_hidden_dim

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_X_PATH = PROJECT_ROOT / "X_train_sparse.npz"          # chemical (drug cocktail)
DEFAULT_Y_PATH = PROJECT_ROOT / "Y_train_sparse.npz"           # clinical (ADR)

# Compact transcriptomic artifacts
DEFAULT_TRANS_PROFILES_PATH = (
    PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_drug_profiles.npy"
)
DEFAULT_TRANS_PATIENT_INDICES_PATH = (
    PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_patient_profile_indices.npy"
)

DEFAULT_CHEM_CHECKPOINT = (
    PROJECT_ROOT / "output" / "autoencoder" / "drug_adr_encoder" / "drug_adr_encoder.pt"
)

MAX_DRUGS = 5
FINGERPRINT_BITS = 3_095
CHEM_DIM = MAX_DRUGS * FINGERPRINT_BITS

BRANCH_EMBED_DIM = 256  # each modality is compressed to this before fusion


def load_sparse_npz_fast(path: Path) -> sparse.csr_matrix:
    """
    scipy.sparse.load_npz's constructor calls get_index_dtype(), which does a full
    .max() scan over the `indices` array to pick a "safe" dtype. At ~5.76B nonzeros
    that scan alone is slow, and combined with the RAM footprint of loading the whole
    matrix (tens of GB even at int32/float32), it can look like a hang rather than
    just a slow load. Building the matrix by assigning arrays directly skips that
    validating constructor path since we already know these dtypes are correct
    (this is the same matrix smiles.py wrote with downcast indices).
    """
    with np.load(path) as loaded:
        data = loaded["data"]
        indices = loaded["indices"]
        indptr = loaded["indptr"]
        shape = tuple(loaded["shape"])

    matrix = sparse.csr_matrix(shape, dtype=data.dtype)
    matrix.data = data
    matrix.indices = indices
    matrix.indptr = indptr
    return matrix


def compute_clin_pos_weight(clin_matrix: sparse.csr_matrix, *, max_weight: float) -> torch.Tensor:
    """
    Per-column pos_weight for BCEWithLogitsLoss: (# negatives / # positives), capped at
    max_weight. Without this, clin_matrix's ~0.014% positive density lets the model hit
    a near-zero loss just by predicting "no ADR" everywhere -- flat loss, no real learning.
    This mirrors the pos_weight_max approach already validated in drug_adr_encoder.py.
    """
    n_rows = clin_matrix.shape[0]
    pos_counts = np.asarray(clin_matrix.sum(axis=0)).ravel()  # nnz per column (values are 0/1)
    pos_counts = np.clip(pos_counts, 1, None)  # avoid divide-by-zero for ADRs with no positives
    neg_counts = n_rows - pos_counts
    pos_weight = np.clip(neg_counts / pos_counts, 1.0, max_weight)
    return torch.from_numpy(pos_weight.astype(np.float32))


class FeedForwardEncoder(nn.Module):
    """Generic dense encoder used for the transcriptomic and clinical branches."""

    def __init__(self, *, in_dim: int, out_dim: int = BRANCH_EMBED_DIM,
                 hidden_dims: tuple[int, int] = (1024, 512), dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultimodalAutoencoder(nn.Module):
    """
    chem/trans/clin -> 3x256 encoders -> concat(768) -> joint hidden(512) ->
    bottleneck z -> decoder hidden(512) -> 2 reconstruction heads.
    """

    def __init__(
        self,
        *,
        trans_dim: int,
        clinical_dim: int,
        chem_slot_embed_dim: int = default_slot_embed_dim,
        chem_hidden_dim: int = default_chemical_hidden_dim,
        bottleneck_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.chem_encoder = DrugEncoder(
            slot_embed_dim=chem_slot_embed_dim,
            hidden_dim=chem_hidden_dim,
            latent_dim=BRANCH_EMBED_DIM,
            dropout=dropout,
        )
        self.trans_encoder = FeedForwardEncoder(in_dim=trans_dim, dropout=dropout)
        self.clin_encoder = FeedForwardEncoder(in_dim=clinical_dim, dropout=dropout)

        fused_dim = BRANCH_EMBED_DIM * 3
        self.joint_hidden = nn.Linear(fused_dim, 512)
        self.bottleneck_z = nn.Linear(512, bottleneck_dim)
        self.decoder_hidden = nn.Linear(bottleneck_dim, 512)

        self.trans_decoder_head = nn.Linear(512, trans_dim)
        self.clin_decoder_head = nn.Linear(512, clinical_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, chem_in: torch.Tensor, trans_in: torch.Tensor, clin_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chem_emb = self.chem_encoder(chem_in)
        trans_emb = self.trans_encoder(trans_in)
        clin_emb = self.clin_encoder(clin_in)

        fused = torch.cat([chem_emb, trans_emb, clin_emb], dim=1)
        joint = self.dropout(self.relu(self.joint_hidden(fused)))
        z = self.relu(self.bottleneck_z(joint))
        decoder_hidden = self.dropout(self.relu(self.decoder_hidden(z)))

        trans_recon = self.trans_decoder_head(decoder_hidden)
        clin_recon = self.clin_decoder_head(decoder_hidden)
        return trans_recon, clin_recon, z

    def load_pretrained_chem_encoder(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        encoder_weights = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        missing, unexpected = self.chem_encoder.load_state_dict(encoder_weights, strict=False)
        print(f"Warm-started chem_encoder from {checkpoint_path}")
        if missing:
            print(f"  missing keys (fine if latent_dim differs): {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")


class TrimodalDataset:
    """
    Holds the row-aligned chemical / clinical sparse matrices and the compact
    transcriptomic profile bank + patient index array. No __getitem__ here on
    purpose -- see BatchedTrimodalLoader below for why.
    """

    def __init__(
        self,
        chemical_matrix: sparse.csr_matrix,
        clinical_matrix: sparse.csr_matrix,
        trans_profiles: np.ndarray,       # mmap'd, shape (n_profiles, genes)
        trans_patient_indices: np.ndarray,  # mmap'd, shape (n_patients, max_drugs)
    ) -> None:
        self.chemical_matrix = chemical_matrix.tocsr()
        self.clinical_matrix = clinical_matrix.tocsr()
        self.trans_profiles = trans_profiles
        self.trans_patient_indices = trans_patient_indices

        if self.chemical_matrix.shape[0] != trans_patient_indices.shape[0]:
            raise ValueError(
                "Chemical and transcriptomic-index row counts do not match: "
                f"{self.chemical_matrix.shape[0]:,} vs {trans_patient_indices.shape[0]:,}. "
                "Did you build X_train_sparse.npz and the transcriptomic patient indices "
                "from the same FAERS file/row order?"
            )
        if self.chemical_matrix.shape[0] != self.clinical_matrix.shape[0]:
            raise ValueError(
                "Chemical and clinical row counts do not match: "
                f"{self.chemical_matrix.shape[0]:,} vs {self.clinical_matrix.shape[0]:,}."
            )

    def __len__(self) -> int:
        return self.chemical_matrix.shape[0]


class BatchedTrimodalLoader:
    """
    Deliberately NOT a torch DataLoader. torch's default DataLoader calls
    dataset[i] one row at a time then collates -- with a CSR matrix at ~5.76B
    nonzeros, a single-row fancy-index lookup is expensive, and doing it 256
    times per batch (57,800 times per epoch) is the actual bottleneck we hit.

    scipy can pull an entire batch of rows out of a CSR matrix in one call
    (matrix[array_of_indices]), which is far cheaper than looping single-row
    lookups. This also sidesteps torch's num_workers/multiprocessing path,
    which on macOS uses 'spawn' and would otherwise pickle a full copy of
    these multi-GB matrices into every worker process at startup.
    """

    def __init__(self, dataset: TrimodalDataset, *, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            batch_idx = order[start : start + self.batch_size]

            chem_batch = torch.from_numpy(
                self.dataset.chemical_matrix[batch_idx].toarray().astype(np.float32, copy=False)
            )
            clin_batch = torch.from_numpy(
                self.dataset.clinical_matrix[batch_idx].toarray().astype(np.float32, copy=False)
            )

            profile_index_batch = self.dataset.trans_patient_indices[batch_idx]  # (batch, max_drugs)
            trans_rows = np.zeros(
                (len(batch_idx), self.dataset.trans_profiles.shape[1]), dtype=np.float32
            )
            valid_counts = np.zeros(len(batch_idx), dtype=np.float32)
            for slot in range(profile_index_batch.shape[1]):
                slot_indices = profile_index_batch[:, slot]
                valid_mask = slot_indices >= 0
                if not valid_mask.any():
                    continue
                trans_rows[valid_mask] += self.dataset.trans_profiles[slot_indices[valid_mask]]
                valid_counts[valid_mask] += 1
            matched_mask = valid_counts > 0
            trans_rows[matched_mask] /= valid_counts[matched_mask, None]
            trans_batch = torch.from_numpy(trans_rows)

            yield chem_batch, trans_batch, clin_batch


class PrefetchLoader:
    """
    Wraps BatchedTrimodalLoader and prepares the NEXT batch on a background thread
    while the main thread runs the model forward/backward on the CURRENT batch.

    Right now those two things happen serially: build batch (CPU, sparse->dense) then
    train on it (MPS/GPU), one after another, batch after batch. Any time spent on
    GPU compute is currently CPU-idle, and vice versa. Overlapping them is close to
    free throughput.

    Deliberately a background *thread*, not a process: threading avoids the
    macOS-spawn pickling problem that ruled out num_workers earlier (no copies of the
    multi-GB matrices get made), and numpy/scipy release the GIL during the large
    array operations this loader does, so a single prefetch thread can genuinely run
    concurrently with the main thread's torch calls.
    """

    def __init__(self, loader: "BatchedTrimodalLoader", *, queue_size: int = 2) -> None:
        self.loader = loader
        self.queue_size = queue_size

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        SENTINEL = object()

        def producer() -> None:
            try:
                for batch in self.loader:
                    q.put(batch)
            finally:
                q.put(SENTINEL)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is SENTINEL:
                break
            yield item


def run_epoch(
    model: MultimodalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    *,
    bce: nn.Module,
    mse: nn.Module,
    loss_weights: tuple[float, float],
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 200,
) -> dict[str, float]:
    import time

    training = optimizer is not None
    model.train(training)

    totals = {"trans": 0.0, "clin": 0.0, "total": 0.0}
    n = 0
    n_batches = len(loader)
    start_time = time.time()
    predicted_positive_count = 0
    true_positive_count = 0
    total_adr_slots = 0

    for batch_idx, (chem_batch, trans_batch, clin_batch) in enumerate(loader, start=1):
        chem_batch = chem_batch.to(device)
        trans_batch = trans_batch.to(device)
        clin_batch = clin_batch.to(device)

        if training:
            optimizer.zero_grad()

        trans_recon, clin_recon, _ = model(chem_batch, trans_batch, clin_batch)

        trans_loss = mse(trans_recon, trans_batch)
        clin_loss = bce(clin_recon, clin_batch)

        w_trans, w_clin = loss_weights
        loss = w_trans * trans_loss + w_clin * clin_loss

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = chem_batch.size(0)
        totals["trans"] += trans_loss.item() * batch_size
        totals["clin"] += clin_loss.item() * batch_size
        totals["total"] += loss.item() * batch_size
        n += batch_size

        with torch.no_grad():
            predicted_positive_count += (torch.sigmoid(clin_recon) > 0.5).sum().item()
            true_positive_count += (clin_batch > 0.5).sum().item()
            total_adr_slots += clin_batch.numel()

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - start_time
            batches_per_sec = batch_idx / elapsed
            eta_sec = (n_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else float("nan")
            pred_rate = predicted_positive_count / total_adr_slots if total_adr_slots else 0.0
            true_rate = true_positive_count / total_adr_slots if total_adr_slots else 0.0
            print(
                f"  batch {batch_idx:,}/{n_batches:,} "
                f"({batches_per_sec:.2f} batch/s, ETA {eta_sec / 60:.1f} min) "
                f"loss={totals['total'] / n:.4f} "
                f"| predicted-positive-rate={pred_rate:.5f} (true-rate={true_rate:.5f})"
            )

    return {
        key: value / max(n, 1) for key, value in totals.items()
    } | {
        "predicted_positive_rate": predicted_positive_count / max(total_adr_slots, 1),
        "true_positive_rate": true_positive_count / max(total_adr_slots, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the trimodal FAERS autoencoder.")
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument("--chem-checkpoint", type=Path, default=DEFAULT_CHEM_CHECKPOINT)
    parser.add_argument("--warm-start-chem", action="store_true")
    parser.add_argument("--freeze-chem", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--trans-loss-weight", type=float, default=1.0)
    parser.add_argument("--clin-loss-weight", type=float, default=1.0)
    parser.add_argument("--clin-pos-weight-max", type=float, default=10.0,
                         help="Cap on per-ADR pos_weight for the clinical BCE loss. "
                              "Without this the model can trivially minimize loss by "
                              "predicting no ADRs, given how sparse clin_matrix is.")
    parser.add_argument("--checkpoint-dir", type=Path,
                         default=PROJECT_ROOT / "output" / "autoencoder" / "multimodal")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                         help="Save a checkpoint every N epochs (1 = every epoch).")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required_path in (args.x_path, args.y_path, args.trans_profiles, args.trans_patient_indices):
        if not required_path.exists():
            raise FileNotFoundError(
                f"{required_path} does not exist. If it's one of the transcriptomic "
                "artifacts, run process_l1000_signatures.py then transcriptomic_data.py first."
            )

    print("Loading chemical matrix (X_train_sparse.npz)...")
    chem_matrix = load_sparse_npz_fast(args.x_path)
    print(f"  chem_matrix: {chem_matrix.shape}, nnz={chem_matrix.nnz:,}")

    print("Loading clinical matrix (Y_train_sparse.npz)...")
    clin_matrix = load_sparse_npz_fast(args.y_path)
    print(f"  clin_matrix: {clin_matrix.shape}, nnz={clin_matrix.nnz:,}")

    print("Loading transcriptomic artifacts...")
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")
    print(f"  trans_profiles: {trans_profiles.shape}, patient_indices: {trans_patient_indices.shape}")

    print("Building dataset...")
    dataset = TrimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)
    loader = PrefetchLoader(BatchedTrimodalLoader(dataset, batch_size=args.batch_size, shuffle=True))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = MultimodalAutoencoder(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=args.dropout,
    ).to(device)

    if args.warm_start_chem:
        model.load_pretrained_chem_encoder(args.chem_checkpoint, device)
        if args.freeze_chem:
            for param in model.chem_encoder.parameters():
                param.requires_grad = False
            print("Chemical encoder frozen.")

    print(f"Computing clinical pos_weight (capped at {args.clin_pos_weight_max})...")
    clin_pos_weight = compute_clin_pos_weight(clin_matrix, max_weight=args.clin_pos_weight_max).to(device)
    print(
        f"  pos_weight range: [{clin_pos_weight.min().item():.2f}, "
        f"{clin_pos_weight.max().item():.2f}], mean={clin_pos_weight.mean().item():.2f}"
    )

    bce = nn.BCEWithLogitsLoss(pos_weight=clin_pos_weight)
    mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_weights = (args.trans_loss_weight, args.clin_loss_weight)

    if args.smoke_test:
        chem_batch, trans_batch, clin_batch = next(iter(loader))
        # Wrap batch item into iterable structure for the engine loop
        test_loader = [(chem_batch, trans_batch, clin_batch)]
        with torch.no_grad():
            losses = run_epoch(
                model, test_loader, device,
                bce=bce, mse=mse, loss_weights=loss_weights,
            )
        print(f"Smoke test passed: {losses}")
        return

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to {args.checkpoint_dir}")

    for epoch in range(1, args.epochs + 1):
        losses = run_epoch(
            model, loader, device, bce=bce, mse=mse,
            loss_weights=loss_weights, optimizer=optimizer,
        )
        print(
            f"Epoch {epoch:02d} | total={losses['total']:.4f} | trans={losses['trans']:.4f} | clin={losses['clin']:.4f} "
            f"| predicted-positive-rate={losses['predicted_positive_rate']:.5f} "
            f"(true-rate={losses['true_positive_rate']:.5f})"
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_path = args.checkpoint_dir / f"multimodal_autoencoder_epoch{epoch:02d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            # Also keep a stable "latest" pointer so you don't have to guess the filename.
            latest_path = args.checkpoint_dir / "latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "args": vars(args),
                },
                latest_path,
            )
            print(f"  saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()