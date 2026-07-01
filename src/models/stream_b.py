"""
Stream B: Periodic Classifier (ExoNet-style extended)

Operates on phase-folded views after TLS period search.
Multi-scale CNN + 8-head MHA on local view for transit morphology learning.
Incorporates stellar parameters and centroid motion for BLEND discrimination.

Architecture:
  global_view (2001pt) → LightCurveCNN → 64-dim
  local_view  (201pt)  → MultiHeadAttentionCNN → 64-dim + attention weights
  centroid    (201pt)  → LightCurveCNN → 32-dim
  stellar     (8-dim)  → MLP → 32-dim
  concat (192) → Fusion Linear → 128-dim feature vector
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ── 1D Convolutional building blocks ─────────────────────────────────────────

class Conv1dBlock(nn.Module):
    """
    1D Convolutional block: Conv → BatchNorm → GELU → MaxPool.
    Dilation allows multi-scale temporal receptive field.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pool: bool = True
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            stride=stride, dilation=dilation, padding=pad
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class LightCurveCNN(nn.Module):
    """
    1D CNN encoder for a phase-folded view.
    4 convolutional blocks with increasing channels and dilations.
    Flattened and projected to out_features dimensions.
    """

    def __init__(self, seq_len: int, out_features: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(
            Conv1dBlock(1, 16, kernel_size=5),
            Conv1dBlock(16, 32, kernel_size=5),
            Conv1dBlock(32, 64, kernel_size=5, dilation=2),
            Conv1dBlock(64, 64, kernel_size=3),
        )
        # Compute output length dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            out_len = self.blocks(dummy).shape[-1]
        self.flatten_dim = 64 * out_len
        self.proj = nn.Linear(self.flatten_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len)
        Returns:
            (B, out_features)
        """
        x = x.unsqueeze(1)         # (B, 1, seq_len)
        x = self.blocks(x)          # (B, 64, out_len)
        x = x.flatten(1)            # (B, 64 * out_len)
        return self.proj(x)         # (B, out_features)


class MultiHeadAttentionCNN(nn.Module):
    """
    CNN feature extractor + Multi-Head Self-Attention over temporal dimension.
    Applied to local view (201pt) to learn transit ingress/egress weighting.

    Returns pooled feature + raw attention weights (for XAI rollout).
    """

    def __init__(self, seq_len: int = 201, d_model: int = 64, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            Conv1dBlock(1, 32, kernel_size=5),
            Conv1dBlock(32, 64, kernel_size=5),
            Conv1dBlock(64, d_model, kernel_size=3),
        )
        # MHA over temporal dimension of CNN output
        self.mha = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len)
        Returns:
            pooled: (B, d_model)
            attn_weights: (B, T', T') attention map for XAI
        """
        x = x.unsqueeze(1)                      # (B, 1, seq_len)
        feat = self.cnn(x)                       # (B, d_model, T')
        feat_t = feat.permute(0, 2, 1)          # (B, T', d_model)
        attended, attn_weights = self.mha(feat_t, feat_t, feat_t)
        attended = self.norm(attended)           # (B, T', d_model)
        attended = attended.permute(0, 2, 1)    # (B, d_model, T')
        pooled = self.pool(attended).squeeze(-1) # (B, d_model)
        return pooled, attn_weights


# ── Stream B: Full Periodic Classifier ───────────────────────────────────────

class StreamB(nn.Module):
    """
    Periodic Classifier stream — Stream B of ECLIPSE-PRIME.

    Encodes 4 input modalities:
      1. global_view (2001pt) — full orbit CNN
      2. local_view  (201pt)  — transit morphology CNN + MHA
      3. centroid    (201pt)  — centroid motion CNN (BLEND discriminator)
      4. stellar     (8-dim)  — star properties MLP

    Concatenates all features and fuses to out_features dimensions.
    """

    def __init__(
        self,
        stellar_dim: int = 8,
        centroid_len: int = 201,
        out_features: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        # Global view encoder (2001 pts → 64-dim)
        self.global_cnn = LightCurveCNN(seq_len=2001, out_features=64)
        # Local view encoder with MHA (201 pts → 64-dim)
        self.local_mha_cnn = MultiHeadAttentionCNN(seq_len=201, d_model=64, nhead=8)
        # Centroid motion encoder (201 pts → 32-dim)
        self.centroid_cnn = LightCurveCNN(seq_len=centroid_len, out_features=32)
        # Stellar parameters MLP (8-dim → 32-dim)
        self.stellar_mlp = nn.Sequential(
            nn.Linear(stellar_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32)
        )
        # Fusion: 64 + 64 + 32 + 32 = 192 → out_features
        self.fusion = nn.Sequential(
            nn.Linear(192, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        stellar_params: torch.Tensor,
        centroid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_view:    (B, 2001)
            local_view:     (B, 201)
            stellar_params: (B, 8)
            centroid:       (B, 201)
        Returns:
            features:       (B, out_features)
            attn_weights:   (B, T', T') from local view MHA (for XAI)
        """
        g = self.global_cnn(global_view)                         # (B, 64)
        l, attn_weights = self.local_mha_cnn(local_view)        # (B, 64), attn
        c = self.centroid_cnn(centroid)                          # (B, 32)
        s = self.stellar_mlp(stellar_params)                     # (B, 32)

        combined = torch.cat([g, l, c, s], dim=-1)              # (B, 192)
        features = self.fusion(combined)                         # (B, out_features)
        return features, attn_weights
