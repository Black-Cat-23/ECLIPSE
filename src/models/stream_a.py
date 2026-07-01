"""
Stream A: Temporal Anomaly Transformer (ExoVeil-inspired)

Operates directly on raw PDCSAP flux (no period search required).
Key advantage: detects SINGLE-TRANSIT events that phase-fold methods miss.
In TESS 27-day sectors, many long-period planets (P > 13d) show only 1 transit.

Architecture:
  Raw flux (T cadences)
    → PatchEmbedding (1D Conv patch projector, d=128)
    → TransformerEncoder (Pre-LN, 4 layers, 8 heads)
    → AdaptiveAvgPool1d
    → Feature vector (B, 128)

Inspired by: ExoVeil (Priyanshu 2026, arXiv:2606.02778)
Implemented from scratch — no pretrained weights used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class PatchEmbedding(nn.Module):
    """
    Embed raw flux into patch tokens for the Transformer.

    Uses a 1D Conv as a patch projector (same as ViT image patches).
    Patch size = 64 cadences (~2.1 hours), stride = 32 (50% overlap).
    Learnable positional embeddings.
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        d_model: int = 128,
        max_seq_len: int = 20000
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        # Conv1d as patch projector: 1 channel → d_model
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        # Maximum number of patches
        n_patches = (max_seq_len - patch_size) // stride + 1
        self.pos_embed = nn.Embedding(n_patches + 2, d_model)  # +2 for safety

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) raw flux, padded to T_max
        Returns:
            (B, n_patches, d_model)
        """
        x = x.unsqueeze(1)          # (B, 1, T)
        x = self.proj(x)            # (B, d_model, n_patches)
        x = x.permute(0, 2, 1)     # (B, n_patches, d_model)
        n = x.shape[1]
        pos = torch.arange(n, device=x.device)
        x = x + self.pos_embed(pos)  # broadcast over batch
        return x


class StreamA(nn.Module):
    """
    Temporal Anomaly Transformer — Stream A of ECLIPSE-PRIME.

    This stream operates on raw PDCSAP flux without requiring a period.
    It is the key component that enables detection of single-transit events
    (long-period planets, highly inclined orbits) that phase-fold methods
    fail on by construction.

    The Transformer learns to distinguish between:
      - Smooth stellar variability (attends globally)
      - Localized transit dips (attends locally, high surprise)
      - Instrument systematics (attends periodically to known patterns)

    Pre-LN (norm_first=True) is used for training stability.
    Gradient checkpointing can be applied per-layer to save VRAM on T4.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        patch_size: int = 64,
        max_seq_len: int = 20000,
        dropout: float = 0.1,
        use_grad_checkpoint: bool = True
    ):
        super().__init__()
        self.use_grad_checkpoint = use_grad_checkpoint

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            stride=patch_size // 2,
            d_model=d_model,
            max_seq_len=max_seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True    # Pre-LN for training stability (removes need for warmup)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # disable for variable-length sequences
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) raw padded flux
        Returns:
            (B, d_model) feature vector
        """
        patches = self.patch_embed(x)   # (B, n_patches, d_model)

        if self.use_grad_checkpoint and self.training:
            # Gradient checkpointing: recompute activations during backward
            # Saves ~40% VRAM at cost of ~30% extra compute
            out = grad_checkpoint(self.transformer, patches, use_reentrant=False)
        else:
            out = self.transformer(patches)

        out = self.norm(out)                    # (B, n_patches, d_model)
        out = out.permute(0, 2, 1)             # (B, d_model, n_patches)
        out = self.pool(out).squeeze(-1)        # (B, d_model)
        return out
