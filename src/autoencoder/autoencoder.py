"""
Multimodal Deep Learning Pipeline for Multi-Drug ADR Prediction

Takes in chemical structures (PubChem fingerprints / drug cocktail vectors) and
transcriptomic profiles (LINCS L1000 gene expression matrices) as inputs.

Transcriptomic side: every LINCS L1000 signature for a drug is encoded
separately, then the per-signature embeddings are pooled (masked mean)
into a single drug-level transcriptomic embedding.

Chemical Encoder + Pooled Transcriptomic Encoder -> Concatenation -> Fused Latent Z -> ADR Prediction Head

Loss:
Weighted Binary Cross-Entropy (BCEWithLogitsLoss) targeting FAERS clinical ADR profiles.
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn

# Reuse validated chemical encoder and dataset split utilities
from drug_adr_encoder import (
    DrugEncoder,
    default_chemical_hidden_dim,
    default_slot_embed_dim,
    choose_top_adverse_event_columns,
    split_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default data paths
DEFAULT_X_PATH = PROJECT_ROOT / "X_train_sparse.npz"        # Chemical inputs
DEFAULT_Y_PATH = PROJECT_ROOT / "Y_train_sparse.npz"        # Clinical targets (ADRs)

DEFAULT_TRANS_PROFILES_PATH = (
    PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_drug_profiles.npy"
)
DEFAULT_TRANS_PATIENT_INDICES_PATH = (
    PROJECT_ROOT / "output" / "transcriptomic" / "transcriptomic_patient_profile_indices.npy"
)

DEFAULT_CHEM_CHECKPOINT = (
    PROJECT_ROOT / "output" / "autoencoder" / "drug_adr_encoder" / "drug_adr_encoder.pt"
)

BRANCH_EMBED_DIM = 512  # Dimension for each branch before concatenation
DEFAULT_TRANS_SCALE = 1.0  # trans_input_norm (BatchNorm) now adapts to raw
# feature scale automatically; this is a leftover manual knob, kept in case
# a specific data revision needs it, but no longer load-bearing for scale.


def load_sparse_npz_fast(path: Path) -> sparse.csr_matrix:
    """Fast, low-memory load for large scipy sparse matrices"""
    with np.load(path) as loaded:
        return sparse.csr_matrix(
            (loaded["data"], loaded["indices"], loaded["indptr"]),
            shape=tuple(loaded["shape"]),
            dtype=np.float32,
        )


def compute_clin_pos_weight(clin_matrix: sparse.csr_matrix, *, max_weight: float) -> torch.Tensor:
    """Computes capped positive weights for BCE loss to handle sparse ADR labels."""
    n_rows = clin_matrix.shape[0]
    pos_counts = np.asarray(clin_matrix.sum(axis=0)).ravel()
    pos_counts = np.clip(pos_counts, 1, None)
    neg_counts = n_rows - pos_counts
    pos_weight = np.clip(neg_counts / pos_counts, 1.0, max_weight)
    return torch.from_numpy(pos_weight.astype(np.float32))


def configure_torch_runtime(device: torch.device) -> None:
    """Enable low-risk runtime settings that improve matrix-heavy training speed."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def get_autocast_context(device: torch.device):
    """Return an autocast context for the current accelerator, or a no-op on CPU."""
    try:
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
    except Exception:
        pass
    return nullcontext()


def csr_matrix_to_torch_csr(matrix: sparse.csr_matrix, *, device: torch.device) -> torch.Tensor:
    """Convert a SciPy CSR matrix to a torch CSR tensor without densifying."""
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64, copy=False)),
        torch.from_numpy(matrix.indices.astype(np.int64, copy=False)),
        torch.from_numpy(matrix.data.astype(np.float32, copy=False)),
        size=matrix.shape,
        device=device,
    )


class FeedForwardEncoder(nn.Module):
    """Dense feed-forward encoder with LeakyReLU to prevent dead unit collapse."""

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int = BRANCH_EMBED_DIM,
        hidden_dims: tuple[int, int] = (1024, 512),
        dropout: float = 0.0,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SparseAwareDrugEncoder(DrugEncoder):
    """Drug encoder that can consume sparse fingerprint batches without densifying."""

    def forward(self, chemical_input):  # type: ignore[override]
        if sparse.issparse(chemical_input):
            return self._forward_sparse_csr(chemical_input.tocsr())
        if isinstance(chemical_input, torch.Tensor) and chemical_input.is_sparse_csr:
            return self._forward_sparse_torch(chemical_input)
        return super().forward(chemical_input)

    def _forward_sparse_torch(self, chemical_input: torch.Tensor) -> torch.Tensor:
        device = self.slot_embedder.linear.weight.device
        if device.type == "mps":
            return super().forward(chemical_input.to_dense())

        slot_embeddings: list[torch.Tensor] = []
        slot_weight = self.slot_embedder.linear.weight.t().float()
        slot_bias = self.slot_embedder.linear.bias

        with torch.autocast(device_type="cuda", enabled=False):
            for slot in range(self.max_drugs):
                start = slot * self.slot_bits
                end = start + self.slot_bits
                slot_sparse = chemical_input[:, start:end].to(device=device, dtype=torch.float32)
                slot_emb = torch.sparse.mm(slot_sparse, slot_weight)
                if slot_bias is not None:
                    slot_emb = slot_emb + slot_bias.float()
                slot_embeddings.append(self.slot_embedder.dropout(self.slot_embedder.activation(slot_emb)))

        flattened_slots = torch.cat(slot_embeddings, dim=1)
        chemical_hidden = self.dropout(self.activation(self.chemical_hidden(flattened_slots)))
        return self.activation(self.latent_layer(chemical_hidden))

    def _forward_sparse_csr(self, chemical_input: sparse.csr_matrix) -> torch.Tensor:
        device = self.slot_embedder.linear.weight.device
        if device.type == "mps":
            return super().forward(torch.from_numpy(chemical_input.toarray().astype(np.float32, copy=False)).to(device))

        slot_embeddings: list[torch.Tensor] = []
        slot_weight = self.slot_embedder.linear.weight.t().float()
        slot_bias = self.slot_embedder.linear.bias

        with torch.autocast(device_type="cuda", enabled=False):
            for slot in range(self.max_drugs):
                start = slot * self.slot_bits
                end = start + self.slot_bits
                slot_matrix = chemical_input[:, start:end]
                slot_sparse = csr_matrix_to_torch_csr(slot_matrix, device=device)
                slot_emb = torch.sparse.mm(slot_sparse, slot_weight)
                if slot_bias is not None:
                    slot_emb = slot_emb + slot_bias.float()
                slot_embeddings.append(self.slot_embedder.dropout(self.slot_embedder.activation(slot_emb)))

        flattened_slots = torch.cat(slot_embeddings, dim=1)
        chemical_hidden = self.dropout(self.activation(self.chemical_hidden(flattened_slots)))
        return self.activation(self.latent_layer(chemical_hidden))


class MultimodalADRPredictor(nn.Module):
    """
    Multimodal neural network for ADR prediction.

    Each drug's LINCS signatures are encoded individually and pooled
    (masked mean) into one Trans embedding.
    Chem (256-d) + Pooled Trans (256-d) -> Concat (512-d) -> Fusion (512-d) -> Latent Z (256-d) -> ADR Head
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

        # Branch Encoders
        self.chem_encoder = SparseAwareDrugEncoder(
            slot_embed_dim=chem_slot_embed_dim,
            hidden_dim=chem_hidden_dim,
            latent_dim=BRANCH_EMBED_DIM,
            dropout=dropout,
        )
        self.trans_encoder = FeedForwardEncoder(in_dim=trans_dim, dropout=dropout)

        # Input-side normalization for the transcriptomic branch. Raw LINCS
        # signatures times --trans-scale were reaching pre-encoder magnitudes
        # in the millions (Pre-norm Range up to ~1.5M), which is what was
        # driving trans_norm's gamma to collapse toward 0 below. BatchNorm
        # here standardizes each gene feature using running batch statistics
        # instead of relying on one hand-picked global scale constant, so it
        # adapts to the data instead of needing --trans-scale tuned by hand.
        self.trans_input_norm = nn.BatchNorm1d(trans_dim, affine=True)

        # Per-branch LayerNorm: bounds each branch's embedding scale before
        # fusion so neither branch can dominate on magnitude alone, and stops
        # the chem branch's unbounded growth seen during training (embedding
        # max grew ~85 -> ~8,700 over 7 epochs without this).
        # elementwise_affine=False: with a learnable gamma, trans_norm found
        # a degenerate shortcut -- shrinking gamma toward 0 (observed
        # mean|gamma| ~0.023 at epoch 7) mutes the branch's contribution to
        # the loss and, in the same move, mutes the gradient flowing back
        # into trans_encoder (grad norm ~0.0005 vs param norm ~340 by epoch
        # 12), which is what froze the encoder instead of training it. A
        # fixed (non-learnable) normalization can't take that shortcut: it
        # only standardizes scale, so any branch-weighting the model needs
        # has to happen in fusion_layer's real, monitorable weights instead.
        self.chem_norm = nn.LayerNorm(BRANCH_EMBED_DIM, elementwise_affine=False)
        self.trans_norm = nn.LayerNorm(BRANCH_EMBED_DIM, elementwise_affine=False)

        # Fusion Layers (2 inputs * 256 = 512 fused dimension)
        fused_dim = BRANCH_EMBED_DIM * 2
        self.fusion_layer = nn.Linear(fused_dim, 512)
        self.bottleneck_z = nn.Linear(512, bottleneck_dim)

        # Single Output Head: ADR Classification
        self.adr_head = nn.Linear(bottleneck_dim, clinical_dim)

        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout)
        
        # Debug toggle
        self.debug_forward = False

    def forward(
        self, chem_in: torch.Tensor, trans_in: torch.Tensor, trans_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Chemical Encoding (256-d)
        chem_emb_raw = self.chem_encoder(chem_in)

        # 1a. Transcriptomic: encode every LINCS signature separately, then
        # pool the per-signature embeddings into one drug-level embedding.
        # trans_in:   (batch, n_slots, trans_dim) -- raw profile per slot
        # trans_mask: (batch, n_slots)            -- 1.0 real signature, 0.0 padding
        batch_size, n_slots, trans_dim = trans_in.shape
        flat_trans = trans_in.reshape(batch_size * n_slots, trans_dim).float()
        flat_trans = self.trans_input_norm(flat_trans)
        flat_sig_emb = self.trans_encoder(flat_trans)
        sig_emb = flat_sig_emb.view(batch_size, n_slots, -1)  # (batch, n_slots, BRANCH_EMBED_DIM)

        slot_weight = trans_mask.unsqueeze(-1).to(sig_emb.dtype)  # (batch, n_slots, 1)
        pooled_sum = (sig_emb * slot_weight).sum(dim=1)
        valid_slot_counts = slot_weight.sum(dim=1).clamp(min=1.0)
        trans_emb_raw = pooled_sum / valid_slot_counts  # (batch, BRANCH_EMBED_DIM)

        # 1b. Normalize each branch so neither dominates the fused vector on
        # scale alone, and so per-branch magnitude can't grow unbounded.
        chem_emb = self.chem_norm(chem_emb_raw)
        trans_emb = self.trans_norm(trans_emb_raw)

        # DIAGNOSTIC PRINT: Check branch embedding stats, pre- and post-norm
        if self.debug_forward:
            with torch.no_grad():
                # Measure representation diversity across the batch dimension
                inter_sample_std = chem_emb.std(dim=0).mean().item()
                print("\n[DEBUG FORWARD - Chem Encoder]")
                if sparse.issparse(chem_in):
                    nonzero_count = chem_in.nnz
                elif isinstance(chem_in, torch.Tensor) and chem_in.is_sparse:
                    nonzero_count = chem_in._nnz()
                else:
                    nonzero_count = (chem_in != 0).sum().item()
                print(f"  Input Shape:       {chem_in.shape} (Nonzero: {nonzero_count})")
                print(f"  Chem Embedding (pre-norm):  Shape={chem_emb_raw.shape}")
                print(f"  Pre-norm Stats:    Mean={chem_emb_raw.mean().item():.4f}, Std={chem_emb_raw.std().item():.4f}")
                print(f"  Pre-norm Range:    Min={chem_emb_raw.min().item():.4f}, Max={chem_emb_raw.max().item():.4f}")
                print(f"  Post-norm Stats:   Mean={chem_emb.mean().item():.4f}, Std={chem_emb.std().item():.4f}")
                print(f"  Diversity (Cross-sample Std across batch, post-norm): {inter_sample_std:.6f}")
                if inter_sample_std < 1e-5:
                    print("  ⚠️ WARNING: Chem encoder embeddings are identical across batch items! (Collapse Detected)")

                trans_inter_sample_std = trans_emb.std(dim=0).mean().item()
                print("[DEBUG FORWARD - Trans Encoder]")
                print(f"  Avg valid signatures/drug (batch): {valid_slot_counts.mean().item():.2f}")
                print(f"  Trans Embedding (pre-norm, pooled): Shape={trans_emb_raw.shape}")
                print(f"  Pre-norm Stats:    Mean={trans_emb_raw.mean().item():.4f}, Std={trans_emb_raw.std().item():.4f}")
                print(f"  Pre-norm Range:    Min={trans_emb_raw.min().item():.4f}, Max={trans_emb_raw.max().item():.4f}")
                print(f"  Post-norm Stats:   Mean={trans_emb.mean().item():.4f}, Std={trans_emb.std().item():.4f}")
                print(f"  Diversity (Cross-sample Std across batch, post-norm): {trans_inter_sample_std:.6f}")
                if trans_inter_sample_std < 1e-5:
                    print("  ⚠️ WARNING: Trans encoder embeddings are identical across batch items! (Collapse Detected)")
            self.debug_forward = False  # Reset flag after 1 print per epoch

        # 2. Concatenate Branch Vectors -> (batch_size, 512)
        fused = torch.cat([chem_emb, trans_emb], dim=1)
        # 3. Fuse & Project to Shared Latent Space z -> (batch_size, 256)
        joint = self.dropout(self.relu(self.fusion_layer(fused)))
        z = self.relu(self.bottleneck_z(joint))
        # 4. Predict Clinical ADR Logits
        adr_logits = self.adr_head(z)

        return adr_logits, z

    def load_pretrained_chem_encoder(self, checkpoint_path: Path, device: torch.device) -> None:
        """Starts the chemical encoder from a pre-trained checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle state_dict key variations safely
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder_weights = {
            key.removeprefix("encoder.").removeprefix("chem_encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.") or key.startswith("chem_encoder.")
        }
        
        if not encoder_weights:
            print("⚠️ WARNING: No keys starting with 'encoder.' or 'chem_encoder.' found in checkpoint!")
            encoder_weights = state_dict

        missing, unexpected = self.chem_encoder.load_state_dict(encoder_weights, strict=False)
        print(f"\n[LOAD CHECKPOINT] Warm-started chem_encoder from {checkpoint_path}")
        print(f"  Loaded keys count: {len(encoder_weights)}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")


class BimodalDataset:

    def __init__(
        self,
        chemical_matrix: sparse.csr_matrix,
        clinical_matrix: sparse.csr_matrix,
        trans_profiles: np.ndarray,
        trans_patient_indices: np.ndarray,
    ) -> None:
        self.chemical_matrix = chemical_matrix.tocsr()
        self.clinical_matrix = clinical_matrix.tocsr()
        self.trans_profiles = trans_profiles
        self.trans_patient_indices = trans_patient_indices

        if self.chemical_matrix.shape[0] != trans_patient_indices.shape[0]:
            raise ValueError("Chemical and transcriptomic-index row counts do not match.")
        if self.chemical_matrix.shape[0] != self.clinical_matrix.shape[0]:
            raise ValueError("Chemical and clinical row counts do not match.")

    def __len__(self) -> int:
        return self.chemical_matrix.shape[0]


class BatchedBimodalLoader:
    def __init__(
        self,
        dataset: BimodalDataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        indices: np.ndarray | None = None,
        trans_scale: float = DEFAULT_TRANS_SCALE,
        debug_first_batch: bool = False,
        chemical_as_sparse: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset)) if indices is None else indices
        self.trans_scale = trans_scale
        self.debug_first_batch = debug_first_batch
        self.chemical_as_sparse = chemical_as_sparse

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = np.random.permutation(self.indices) if self.shuffle else self.indices

        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]

            batch_csr = self.dataset.chemical_matrix[batch_idx]
            # Diagnostic checks on input batch formatting
            if self.debug_first_batch:
                print("\n[DEBUG DATA BATCH]")
                print(f"  Batch size: {len(batch_idx)}")
                print(f"  Chem batch shape: {batch_csr.shape}")
                print(f"  Chem nnz: {batch_csr.nnz:,}")
                print(f"  Chem non-zero counts per item (first 5): {np.diff(batch_csr.indptr)[:5].tolist()}")
                if batch_csr.shape[0] > 1:
                    sample_diff = np.abs(batch_csr[0].toarray() - batch_csr[1].toarray()).sum()
                    print(f"  L1 Diff between Drug 0 and Drug 1 in batch: {sample_diff:.2f}")
                    if sample_diff == 0:
                        print("  ⚠️ WARNING: Batch items 0 and 1 are identical!")
                self.debug_first_batch = False

            clin_batch = torch.from_numpy(
                self.dataset.clinical_matrix[batch_idx].toarray().astype(np.float32, copy=False)
            )

            profile_index_batch = self.dataset.trans_patient_indices[batch_idx]
            trans_dim = self.dataset.trans_profiles.shape[1]

            # Keep every LINCS signature separate: (batch, n_slots, trans_dim)
            # plus a mask marking which slots are real vs. padding. The raw
            # profiles are no longer averaged together here -- the model
            # encodes each signature on its own and pools the embeddings.
            batch_size, n_slots = profile_index_batch.shape
            trans_slots = np.zeros((batch_size, n_slots, trans_dim), dtype=np.float16)
            flat_profile_indices = profile_index_batch.reshape(-1)
            valid_mask = flat_profile_indices >= 0
            if valid_mask.any():
                flat_slots = trans_slots.reshape(-1, trans_dim)
                flat_slots[valid_mask] = self.dataset.trans_profiles[flat_profile_indices[valid_mask]].astype(np.float16, copy=False)
            trans_mask = valid_mask.reshape(batch_size, n_slots)

            trans_slots *= self.trans_scale
            trans_batch = torch.from_numpy(trans_slots)
            trans_mask_batch = torch.from_numpy(trans_mask)

            chem_batch = batch_csr if self.chemical_as_sparse else torch.from_numpy(
                batch_csr.toarray().astype(np.float32, copy=False)
            )

            yield chem_batch, trans_batch, trans_mask_batch, clin_batch


class PrefetchLoader:
    """Background prefetching for batch loading."""

    def __init__(self, loader: BatchedBimodalLoader, *, queue_size: int = 2) -> None:
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
    model: MultimodalADRPredictor,
    loader: PrefetchLoader,
    device: torch.device,
    *,
    bce: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 200,
    epoch_num: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    
    # Trigger 1 diagnostic forward print at the start of epoch
    model.debug_forward = True

    total_loss = 0.0
    n_samples = 0
    n_batches = len(loader)
    start_time = time.time()
    
    predicted_positives = 0
    true_positives = 0
    total_slots = 0

    for batch_idx, (chem_batch, trans_batch, trans_mask_batch, clin_batch) in enumerate(loader, start=1):
        non_blocking = device.type == "cuda"
        trans_batch = trans_batch.to(device, non_blocking=non_blocking)
        trans_mask_batch = trans_mask_batch.to(device, non_blocking=non_blocking)
        clin_batch = clin_batch.to(device, non_blocking=non_blocking)

        if isinstance(chem_batch, torch.Tensor):
            chem_batch = chem_batch.to(device, non_blocking=non_blocking)
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context, get_autocast_context(device):
            # Forward pass: Predict ADR logits from chemical + transcriptomic inputs
            adr_logits, _ = model(chem_batch, trans_batch, trans_mask_batch)

            loss = bce(adr_logits.float(), clin_batch.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered at epoch {epoch_num}, batch {batch_idx}. "
                    "This is usually caused by mixed-precision overflow or unstable optimizer state."
                )

        if training:
            loss.backward()

            # DIAGNOSTIC PRINT: Gradient audit on step 1 of training
            if batch_idx == 1:
                grad_norms = []
                param_norms = []
                for name, p in model.chem_encoder.named_parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                        param_norms.append(p.norm().item())

                print(f"\n[DEBUG GRADIENTS - Epoch {epoch_num} Batch 1]")
                if grad_norms:
                    print(f"  Chem Encoder Avg Param Norm: {np.mean(param_norms):.4f}")
                    print(f"  Chem Encoder Avg Grad Norm:  {np.mean(grad_norms):.6f}")
                    print(f"  Chem Encoder Max Grad Norm:  {np.max(grad_norms):.6f}")
                    if np.max(grad_norms) == 0:
                        print("  ⚠️ WARNING: All chemical encoder gradients are 0.0! (Encoder Is Not Learning)")
                else:
                    print("  ⚠️ WARNING: No gradients calculated for chemical encoder! (Check if frozen)")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = chem_batch.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        with torch.no_grad():
            predicted_positives += (torch.sigmoid(adr_logits.float()) > 0.5).sum().item()
            true_positives += (clin_batch > 0.5).sum().item()
            total_slots += clin_batch.numel()

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - start_time
            batches_per_sec = batch_idx / elapsed
            eta_sec = (n_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0.0
            pred_rate = predicted_positives / total_slots if total_slots else 0.0
            true_rate = true_positives / total_slots if total_slots else 0.0

            print(
                f"  batch {batch_idx:,}/{n_batches:,} "
                f"({batches_per_sec:.2f} batch/s, ETA {eta_sec / 60:.1f} min) "
                f"loss={total_loss / n_samples:.4f} "
                f"| pred_pos_rate={pred_rate:.5f} (true_rate={true_rate:.5f})"
            )

    return {
        "loss": total_loss / max(n_samples, 1),
        "predicted_positive_rate": predicted_positives / max(total_slots, 1),
        "true_positive_rate": true_positives / max(total_slots, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal ADR prediction model.")
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--trans-profiles", type=Path, default=DEFAULT_TRANS_PROFILES_PATH)
    parser.add_argument("--trans-patient-indices", type=Path, default=DEFAULT_TRANS_PATIENT_INDICES_PATH)
    parser.add_argument("--chem-checkpoint", type=Path, default=DEFAULT_CHEM_CHECKPOINT)
    parser.add_argument(
        "--top-k-adrs",
        type=int,
        default=83,
        help="Train only on the top-K ADR columns by frequency. Use 0 or a negative value to keep all columns.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows after filtering to examples with at least one selected ADR.",
    )
    parser.add_argument("--freeze-chem", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--trans-scale", type=float, default=DEFAULT_TRANS_SCALE)
    parser.add_argument("--clin-pos-weight-max", type=float, default=20.0)
    parser.add_argument(
        "--row-indices",
        type=Path,
        default=None,
        help="Optional row-index file matching a subset X/Y pair, usually X_train_subset.row_indices.npy.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "output" / "models" / "adr_predictor")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Path to a checkpoint (e.g. latest.pt) to resume model + optimizer state + epoch count from.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--warm-start-chem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm start the chemical encoder from a pre-trained checkpoint (default: True)."
    )

    return parser.parse_args()


def print_chem_encoder_weight_summary(model: MultimodalADRPredictor, tag: str = "Initial") -> None:
    """Helper to audit parameters of the chemical encoder."""
    print(f"\n=== Chemical Encoder Parameter Summary ({tag}) ===")
    for name, p in model.chem_encoder.named_parameters():
        print(
            f"  {name:<35} | Mean: {p.mean().item():10.6f} | "
            f"Std: {p.std().item():10.6f} | RequiresGrad: {p.requires_grad}"
        )


def main() -> None:
    args = parse_args()

    for path in (args.x_path, args.y_path, args.trans_profiles, args.trans_patient_indices):
        if not path.exists():
            raise FileNotFoundError(f"Required data file missing: {path}")

    print("Loading datasets...")
    chem_matrix = load_sparse_npz_fast(args.x_path)
    clin_matrix = load_sparse_npz_fast(args.y_path)
    trans_profiles = np.load(args.trans_profiles, mmap_mode="r")
    trans_patient_indices = np.load(args.trans_patient_indices, mmap_mode="r")

    row_indices_path = args.row_indices
    if row_indices_path is None:
        candidate = args.x_path.with_suffix("").with_suffix(".row_indices.npy")
        if candidate.exists():
            row_indices_path = candidate
    if row_indices_path is not None:
        row_indices = np.load(row_indices_path, mmap_mode="r")
        if chem_matrix.shape[0] != len(row_indices) or clin_matrix.shape[0] != len(row_indices):
            raise ValueError(
                "Subset row indices do not match X/Y row counts. "
                "Make sure the row_indices file was generated with the same subset files."
            )
        chem_matrix = chem_matrix[row_indices]
        clin_matrix = clin_matrix[row_indices]
        trans_patient_indices = trans_patient_indices[row_indices]
        print(f"  Applied subset row indices from {row_indices_path}.\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using Compute Device: {device}")
    configure_torch_runtime(device)

    if args.top_k_adrs > 0:
        selected_columns, selected_counts = choose_top_adverse_event_columns(clin_matrix, args.top_k_adrs)
        clin_matrix = clin_matrix[:, selected_columns].tocsr()
        print(
            f"\nSelected top {len(selected_columns):,} ADR columns "
            f"(count range {int(selected_counts.min())} to {int(selected_counts.max())})."
        )
    else:
        selected_columns = np.arange(clin_matrix.shape[1], dtype=np.int64)
        selected_counts = np.asarray(clin_matrix.sum(axis=0)).ravel().astype(np.int64, copy=False)
        print(f"\nUsing all {len(selected_columns):,} ADR columns.")

    full_target_rows = np.asarray(clin_matrix.sum(axis=1)).ravel() > 0
    qualifying_rows = np.flatnonzero(full_target_rows)
    print(
        f"Rows with at least one selected ADR: {len(qualifying_rows):,} / {clin_matrix.shape[0]:,}"
    )

    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("--max-rows must be at least 1 when provided.")
        if args.max_rows < len(qualifying_rows):
            rng = np.random.default_rng(args.split_seed)
            selected_row_indices = rng.choice(qualifying_rows, size=args.max_rows, replace=False)
            selected_row_indices.sort()
            print(f"Subsampled to {len(selected_row_indices):,} rows for training.")
        else:
            selected_row_indices = qualifying_rows
            print(f"Using all {len(selected_row_indices):,} qualifying rows for training.")
    else:
        selected_row_indices = qualifying_rows
        print(f"Using all {len(selected_row_indices):,} qualifying rows for training.")

    chem_matrix = chem_matrix[selected_row_indices]
    clin_matrix = clin_matrix[selected_row_indices]
    trans_patient_indices = trans_patient_indices[selected_row_indices]

    # Fast, memory-free non-zero check using CSR indptr on the filtered rows.
    non_zero_per_row = np.diff(chem_matrix.indptr)
    zero_rows = np.where(non_zero_per_row == 0)[0]
    n_zero_rows = len(zero_rows)
    valid_indices = np.where(non_zero_per_row > 0)[0]

    print(f"\n[DATA AUDIT - Filtered Chemical Matrix]")
    print(f"  Total samples: {len(non_zero_per_row):,}")
    print(f"  Empty (all-zero) chemical vectors: {n_zero_rows:,} ({n_zero_rows / len(non_zero_per_row):.2%})")
    print(f"  Retaining {len(valid_indices):,} non-empty chemical samples for training...\n")

    dataset = BimodalDataset(chem_matrix, clin_matrix, trans_profiles, trans_patient_indices)

    train_rel, val_rel = split_indices(
        len(valid_indices), args.validation_fraction, args.split_seed
    )
    
    # 2. Index into valid_indices to get real matrix row numbers
    train_indices = valid_indices[train_rel]
    val_indices = valid_indices[val_rel]

    print(f"Dataset Split: {len(train_indices):,} Train / {len(val_indices):,} Validation")
    # Dense chemical batches are usually faster on modern accelerators and
    # still fit comfortably in 16 GB once the transcriptomic buffer is half precision.
    chemical_as_sparse = False
    train_loader = PrefetchLoader(
        BatchedBimodalLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             indices=train_indices, trans_scale=args.trans_scale, debug_first_batch=True,
                             chemical_as_sparse=chemical_as_sparse)
    )
    val_loader = PrefetchLoader(
        BatchedBimodalLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             indices=val_indices, trans_scale=args.trans_scale,
                             chemical_as_sparse=chemical_as_sparse)
    )

    model = MultimodalADRPredictor(
        trans_dim=trans_profiles.shape[1],
        clinical_dim=clin_matrix.shape[1],
        dropout=args.dropout,
    ).to(device)

    print_chem_encoder_weight_summary(model, tag="Initialized")

    if args.warm_start_chem:
        model.load_pretrained_chem_encoder(args.chem_checkpoint, device)
        print_chem_encoder_weight_summary(model, tag="Post Warm-Start")
        if args.freeze_chem:
            for param in model.chem_encoder.parameters():
                param.requires_grad = False
            print("\n🔒 Chemical encoder parameters FROZEN.")

    clin_pos_weight = compute_clin_pos_weight(clin_matrix, max_weight=args.clin_pos_weight_max).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=clin_pos_weight)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        resume_checkpoint = torch.load(args.resume_from, map_location=device)
        load_result = model.load_state_dict(resume_checkpoint["model_state_dict"], strict=False)
        if load_result.missing_keys:
            print(f"  ⚠️  Missing keys (not in checkpoint, using fresh init): {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  ⚠️  Unexpected keys (in checkpoint, dropped): {load_result.unexpected_keys}")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_loss = resume_checkpoint.get("val_metrics", {}).get("loss", float("inf"))
        print(f"  Resuming at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")

    if args.smoke_test:
        print("\nRunning quick smoke test...")
        chem_b, trans_b, trans_mask_b, clin_b = next(iter(train_loader))
        test_loader = [(chem_b, trans_b, trans_mask_b, clin_b)]
        results = run_epoch(model, test_loader, device, bce=bce, epoch_num=1)
        print(f"\nSmoke test completed successfully: {results}")
        return

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting Model Training...")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch:02d}/{args.epochs:02d} ---")
        train_metrics = run_epoch(model, train_loader, device, bce=bce, optimizer=optimizer, epoch_num=epoch)
        
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, bce=bce, optimizer=None, epoch_num=epoch)

        is_best = val_metrics["loss"] < best_val_loss
        best_val_loss = min(best_val_loss, val_metrics["loss"])

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}"
            f"{'  <- Best Val Loss' if is_best else ''}"
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": vars(args),
                "selected_adr_columns": selected_columns,
                "selected_adr_counts": selected_counts,
            }
            torch.save(checkpoint_payload, args.checkpoint_dir / f"adr_predictor_epoch{epoch:02d}.pt")
            torch.save(checkpoint_payload, args.checkpoint_dir / "latest.pt")
            if is_best:
                torch.save(checkpoint_payload, args.checkpoint_dir / "best_val.pt")

    np.save(args.checkpoint_dir / "selected_adr_columns.npy", selected_columns)
    np.save(args.checkpoint_dir / "selected_adr_counts.npy", selected_counts)


if __name__ == "__main__":
    main()
