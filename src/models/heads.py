"""
Multi-Task Prediction Heads for ECLIPSE-PRIME.

Three heads on top of the fused 256-dim representation:

1. ClassificationHead: 4-way softmax
   Classes: TRANSIT (0), EB (1), BLEND (2), OTHER (3)

2. ParameterRegressionHead: Transit parameter estimation with aleatoric uncertainty.
   Outputs: [P_mean, P_logvar, τ_mean, τ_logvar, δ_mean, δ_logvar]
   Uses Gaussian NLL loss: learns both the mean AND uncertainty in one pass.
   Aleatoric uncertainty = irreducible (data noise, transit shape degeneracy).

3. SNRHead: Single scalar regression for transit SNR (Signal Detection Efficiency).
   Softplus activation ensures SNR > 0.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    4-class classification head.
    Returns logits (for Focal Loss) and softmax probabilities.
    """

    def __init__(self, d_fused: int = 256, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_fused, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


class ParameterRegressionHead(nn.Module):
    """
    Transit parameter regression with aleatoric uncertainty.

    For each of the 3 parameters (period P, duration τ, depth δ):
      - Predicts mean value
      - Predicts log_var (log of variance) → learned uncertainty
      - Loss: Gaussian NLL = 0.5 * (log_var + (target - mean)² / var)

    This is the "probabilistic regression" approach from Kendall & Gal (2017).
    The model learns WHEN to be uncertain, not just the mean prediction.
    """

    def __init__(self, d_fused: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_fused, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            # 6 outputs: [P_mean, P_logvar, τ_mean, τ_logvar, δ_mean, δ_logvar]
            nn.Linear(128, 6)
        )
        # Initialize log_var heads to predict zero variance initially
        # (final linear layer bias ≈ 0 → var = exp(0) = 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        # Clamp log_var to prevent exploding variance
        log_var = torch.clamp(out[:, [1, 3, 5]], -10.0, 5.0)
        return {
            "period_mean":     out[:, 0],
            "period_logvar":   log_var[:, 0],
            "duration_mean":   out[:, 2],
            "duration_logvar": log_var[:, 1],
            "depth_mean":      out[:, 4],
            "depth_logvar":    log_var[:, 2],
        }


class SNRHead(nn.Module):
    """
    SNR regression head.
    Softplus ensures output > 0 (SNR is always positive).
    """

    def __init__(self, d_fused: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_fused, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()   # guarantees SNR > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class MultiTaskHead(nn.Module):
    """
    Combined multi-task prediction head.
    Wraps ClassificationHead + ParameterRegressionHead + SNRHead.
    """

    def __init__(self, d_fused: int = 256, n_classes: int = 4):
        super().__init__()
        self.clf = ClassificationHead(d_fused, n_classes)
        self.params = ParameterRegressionHead(d_fused)
        self.snr = SNRHead(d_fused)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, d_fused) — fused representation from CrossAttentionFusion
        Returns:
            dict with keys:
                logits, probs                    — classification
                period_mean, period_logvar        — period P with uncertainty
                duration_mean, duration_logvar    — duration τ with uncertainty
                depth_mean, depth_logvar          — depth δ with uncertainty
                snr_pred                          — predicted SNR
        """
        clf_out = self.clf(x)
        param_out = self.params(x)
        snr_out = self.snr(x)

        return {
            **clf_out,
            **param_out,
            "snr_pred": snr_out
        }
