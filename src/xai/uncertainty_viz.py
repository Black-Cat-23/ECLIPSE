"""
Uncertainty decomposition visualization.
Separates aleatoric (data) and epistemic (model) uncertainty.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def decompose_uncertainty(
    period_mean: float,
    period_logvar: float,
    period_epistemic_std: float,
    duration_mean: float,
    duration_logvar: float,
    duration_epistemic_std: float,
    depth_mean: float,
    depth_logvar: float,
    depth_epistemic_std: float,
    prob_mean: np.ndarray,
    prob_epistemic_std: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Decompose total uncertainty into aleatoric and epistemic components.

    Aleatoric: irreducible data noise (from log_var outputs)
    Epistemic: model uncertainty (from MC Dropout variance)
    Total: sqrt(aleatoric² + epistemic²)

    Returns nested dict with per-parameter breakdown.
    """
    def decompose(aleatoric_std: float, epistemic_std: float) -> dict:
        total = float(np.sqrt(aleatoric_std**2 + epistemic_std**2))
        return {
            "aleatoric": aleatoric_std,
            "epistemic": epistemic_std,
            "total": total,
            "aleatoric_frac": aleatoric_std / total if total > 0 else 0.5
        }

    return {
        "period": decompose(
            float(np.exp(0.5 * period_logvar)), period_epistemic_std
        ),
        "duration": decompose(
            float(np.exp(0.5 * duration_logvar)), duration_epistemic_std
        ),
        "depth": decompose(
            float(np.exp(0.5 * depth_logvar)), depth_epistemic_std
        ),
        "classification": {
            "epistemic_std": float(prob_epistemic_std.mean()),
            "max_class_uncertainty": float(prob_epistemic_std.max())
        }
    }
