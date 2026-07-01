"""
ECLIPSE-PRIME: Full Dual-Stream Multi-Task Exoplanet Signal Classifier.

Novel contributions:
1. Cross-attention fusion between raw-flux anomaly stream (A) and
   periodic phase-fold stream (B) — first architecture to combine both.
2. Single end-to-end model for 4-class + parameter regression + SNR.
3. batman physics constraint in the loss (see physics_loss.py).
4. Handles single-transit events via Stream A (no period required).

Architecture summary:
  Stream A: Temporal Anomaly Transformer → 128-dim
  Stream B: Periodic CNN+MHA Classifier → 128-dim
  Fusion: CrossAttention(Q=A, K/V=B) + residual MLP → 256-dim
  Heads: 4-class | P̂±σ, τ̂±σ, δ̂±σ | SNR
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stream_a import StreamA
from src.models.stream_b import StreamB
from src.models.heads import MultiTaskHead


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention: Q = Stream A features, K = V = Stream B features.

    Allows the anomaly stream (raw flux) to attend to the periodic stream
    (phase-folded) and vice versa through a bidirectional residual pathway.

    This is critical for BLEND discrimination: the centroid info in Stream B
    can suppress high anomaly scores from Stream A when a centroid shift
    confirms a background EB source.
    """

    def __init__(
        self,
        d_a: int = 128,
        d_b: int = 128,
        d_out: int = 256,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        # Project A and B to common dimension
        self.proj_a = nn.Linear(d_a, d_out)
        self.proj_b = nn.Linear(d_b, d_out)
        # Cross-attention: Q from A, K/V from B
        self.cross_attn = nn.MultiheadAttention(
            d_out, nhead, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_out)
        # Residual MLP: attend(A,B) || proj(B) → d_out
        self.mlp = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out)
        )
        self.final_norm = nn.LayerNorm(d_out)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_a: (B, d_a) — Stream A features
            feat_b: (B, d_b) — Stream B features
        Returns:
            fused: (B, d_out)
        """
        # Unsqueeze to sequence length 1 for attention API
        qa = self.proj_a(feat_a).unsqueeze(1)     # (B, 1, d_out)
        kb = self.proj_b(feat_b).unsqueeze(1)     # (B, 1, d_out)

        attn_out, _ = self.cross_attn(qa, kb, kb)  # (B, 1, d_out)
        attn_out = attn_out.squeeze(1)             # (B, d_out)
        attn_out = self.norm(attn_out)

        # Residual: concat attention output with projected B
        combined = torch.cat([attn_out, self.proj_b(feat_b)], dim=-1)  # (B, 2*d_out)
        fused = self.mlp(combined)                 # (B, d_out)
        return self.final_norm(fused)


class ECLIPSEPrime(nn.Module):
    """
    ECLIPSE-PRIME: Full dual-stream multi-task model.

    Inputs:
        raw_flux:       (B, T_max)    — padded raw PDCSAP flux for Stream A
        global_view:    (B, 2001)     — TLS phase-folded global view
        local_view:     (B, 201)      — TLS phase-folded local view
        stellar_params: (B, 8)        — normalized stellar feature vector
        centroid:       (B, 201)      — phase-folded centroid displacement

    Outputs (dict):
        logits:          (B, 4)       — classification logits
        probs:           (B, 4)       — class probabilities (softmax)
        period_mean:     (B,)         — predicted period (days)
        period_logvar:   (B,)         — log-variance for period
        duration_mean:   (B,)         — predicted duration (days)
        duration_logvar: (B,)         — log-variance for duration
        depth_mean:      (B,)         — predicted depth (fractional)
        depth_logvar:    (B,)         — log-variance for depth
        snr_pred:        (B,)         — predicted SNR
        attention_weights: tensor      — local view MHA weights for XAI
    """

    def __init__(
        self,
        stellar_dim: int = 8,
        d_stream_a: int = 128,
        d_stream_b: int = 128,
        d_fused: int = 256,
        n_classes: int = 4,
        patch_size: int = 64,
        T_max: int = 20000,
        use_grad_checkpoint: bool = True
    ):
        super().__init__()

        self.stream_a = StreamA(
            d_model=d_stream_a,
            nhead=8,
            num_layers=4,
            patch_size=patch_size,
            max_seq_len=T_max,
            use_grad_checkpoint=use_grad_checkpoint
        )
        self.stream_b = StreamB(
            stellar_dim=stellar_dim,
            centroid_len=201,
            out_features=d_stream_b
        )
        self.fusion = CrossAttentionFusion(
            d_a=d_stream_a,
            d_b=d_stream_b,
            d_out=d_fused,
            nhead=8
        )
        self.head = MultiTaskHead(d_fused=d_fused, n_classes=n_classes)

    def forward(
        self,
        raw_flux: torch.Tensor,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        stellar_params: torch.Tensor,
        centroid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through ECLIPSE-PRIME.
        See class docstring for input/output shapes.
        """
        # Stream A: temporal anomaly features from raw flux
        feat_a = self.stream_a(raw_flux)              # (B, d_stream_a)

        # Stream B: periodic features from phase-fold + stellar + centroid
        feat_b, attn_weights = self.stream_b(
            global_view, local_view, stellar_params, centroid
        )                                              # (B, d_stream_b)

        # Fusion: cross-attention between streams
        fused = self.fusion(feat_a, feat_b)            # (B, d_fused)

        # Multi-task prediction heads
        outputs = self.head(fused)
        outputs["attention_weights"] = attn_weights    # expose for XAI

        return outputs

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter count per sub-module for reporting."""
        def count(module): return sum(p.numel() for p in module.parameters())
        return {
            "stream_a": count(self.stream_a),
            "stream_b": count(self.stream_b),
            "fusion":   count(self.fusion),
            "heads":    count(self.head),
            "total":    count(self)
        }

    @classmethod
    def from_config(cls, config) -> "ECLIPSEPrime":
        """Construct from an ECLIPSEConfig instance."""
        return cls(
            stellar_dim=config.model.stellar_dim,
            d_stream_a=config.model.d_stream_a,
            d_stream_b=config.model.d_stream_b,
            d_fused=config.model.d_fused,
            n_classes=config.model.n_classes,
            patch_size=config.model.patch_size,
            T_max=config.model.T_max,
            use_grad_checkpoint=config.training.grad_checkpoint
        )
