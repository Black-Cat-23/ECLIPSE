"""
Multi-task loss for ECLIPSE-PRIME.

L_total = 1.0 × Focal_CE(4-class)
        + 0.5 × GaussianNLL(period)
        + 0.5 × GaussianNLL(duration)
        + 0.5 × GaussianNLL(depth)
        + 0.3 × MSE(SNR)
        + 0.2 × PhysicsConstraint(batman)

Labels: class ∈ {0:TRANSIT, 1:EB, 2:BLEND, 3:OTHER}
Parameter + SNR losses computed only for TRANSIT-class samples (has_params=True).
Physics loss similarly TRANSIT-only.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.physics_loss import batman_physics_loss

CLASS_NAMES = ["TRANSIT", "EB", "BLEND", "OTHER"]


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    n_classes: int = 4
) -> torch.Tensor:
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    γ=2.0 reduces the loss contribution from easy examples and focuses
    training on hard, misclassified examples. Critical for the imbalanced
    TRANSIT:OTHER distribution.

    Args:
        logits:   (B, n_classes) — raw model outputs
        targets:  (B,) — class indices
        gamma:    focusing parameter (default 2.0)
        alpha:    (n_classes,) class weights for imbalance correction
        n_classes: number of classes

    Returns:
        scalar focal loss
    """
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)                         # probability of true class
    focal = ((1.0 - pt) ** gamma) * ce
    return focal.mean()


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood loss for probabilistic regression.

    NLL = 0.5 × (log_var + (target - mean)² / exp(log_var))

    When mask is provided, loss is computed only on masked samples.
    Returns 0.0 (non-backpropagated) if no masked samples exist.

    Args:
        mean:    (B,) predicted mean
        log_var: (B,) predicted log-variance (learned uncertainty)
        target:  (B,) ground truth value
        mask:    (B,) bool — compute loss only where True

    Returns:
        scalar Gaussian NLL loss
    """
    if mask is not None and mask.sum() == 0:
        return torch.tensor(0.0, device=mean.device, requires_grad=True)

    var = torch.exp(log_var) + 1e-6        # numerical stability
    nll = 0.5 * (log_var + (target - mean) ** 2 / var)

    if mask is not None:
        nll = nll[mask]

    return nll.mean()


class ECLIPSEMultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for ECLIPSE-PRIME training.

    Weights:
        cls:     1.0  (always computed)
        period:  0.5  (TRANSIT only)
        dur:     0.5  (TRANSIT only)
        depth:   0.5  (TRANSIT only)
        snr:     0.3  (TRANSIT only)
        physics: 0.2  (TRANSIT only, batman consistency)

    class_weights should be a FloatTensor of shape (4,) computed
    from sklearn.utils.class_weight.compute_class_weight('balanced').
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        n_classes: int = 4,
        focal_gamma: float = 2.0,
        physics_weight: float = 0.2
    ):
        super().__init__()
        self.class_weights = class_weights
        self.n_classes = n_classes
        self.focal_gamma = focal_gamma
        self.physics_weight = physics_weight

        # Loss weights
        self.w_cls = 1.0
        self.w_period = 0.5
        self.w_duration = 0.5
        self.w_depth = 0.5
        self.w_snr = 0.3

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            outputs: ECLIPSEPrime.forward() output dict
            targets: {
                'class':      LongTensor (B,) in {0,1,2,3}
                'period':     FloatTensor (B,) in days
                'duration':   FloatTensor (B,) in days
                'depth':      FloatTensor (B,) fractional
                'snr':        FloatTensor (B,) TLS SDE
                'has_params': BoolTensor (B,) True for TRANSIT with known params
                'local_view': FloatTensor (B, 201) for physics loss
            }

        Returns:
            dict with keys: total, cls, period, duration, depth, snr, physics
        """
        device = outputs["logits"].device
        transit_mask = targets.get("has_params",
                                   (targets["class"] == 0))  # fallback: class=0

        # ── 1. Focal classification loss (all samples) ────────────────────────
        alpha = (self.class_weights.to(device)
                 if self.class_weights is not None else None)
        l_cls = focal_loss(
            outputs["logits"], targets["class"],
            gamma=self.focal_gamma, alpha=alpha, n_classes=self.n_classes
        )

        # ── 2. Parameter regression losses (TRANSIT only) ─────────────────────
        l_period = gaussian_nll_loss(
            outputs["period_mean"], outputs["period_logvar"],
            targets["period"], mask=transit_mask
        )
        l_duration = gaussian_nll_loss(
            outputs["duration_mean"], outputs["duration_logvar"],
            targets["duration"], mask=transit_mask
        )
        l_depth = gaussian_nll_loss(
            outputs["depth_mean"], outputs["depth_logvar"],
            targets["depth"], mask=transit_mask
        )

        # ── 3. SNR regression loss (TRANSIT only) ─────────────────────────────
        l_snr = torch.tensor(0.0, device=device)
        if transit_mask.sum() > 0:
            l_snr = F.mse_loss(
                outputs["snr_pred"][transit_mask],
                targets["snr"][transit_mask]
            )

        # ── 4. Batman physics consistency loss (TRANSIT only) ─────────────────
        l_physics = torch.tensor(0.0, device=device)
        if "local_view" in targets and transit_mask.sum() > 0:
            l_physics = batman_physics_loss(
                period_mean=outputs["period_mean"],
                duration_mean=outputs["duration_mean"],
                depth_mean=outputs["depth_mean"],
                local_view_target=targets["local_view"],
                transit_mask=transit_mask,
                device=device
            )

        # ── Total weighted loss ───────────────────────────────────────────────
        l_total = (
            self.w_cls     * l_cls +
            self.w_period  * l_period +
            self.w_duration * l_duration +
            self.w_depth   * l_depth +
            self.w_snr     * l_snr +
            self.physics_weight * l_physics
        )

        return {
            "total":    l_total,
            "cls":      l_cls,
            "period":   l_period,
            "duration": l_duration,
            "depth":    l_depth,
            "snr":      l_snr,
            "physics":  l_physics,
        }
