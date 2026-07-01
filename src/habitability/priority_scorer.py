"""
Observation Priority Scorer & Tier Assignment.

Combines ESI, HZ classification, SNR, and stellar properties
into a single priority score and tier label.

Tier 1 (Highest Priority): ESI > 0.8, CONSERVATIVE HZ, SNR > 10
Tier 2 (High Priority):    ESI > 0.6, any HZ, SNR > 7
Tier 3 (Candidate):        All other TRANSIT-class detections

Priority score is a weighted combination:
    P = 0.40 × ESI + 0.25 × hz_bonus + 0.20 × snr_score + 0.15 × brightness_score
"""
from __future__ import annotations
import numpy as np
from typing import Optional


HZ_BONUS = {
    "CONSERVATIVE": 1.0,
    "INNER":        0.6,
    "OUTER":        0.6,
    "NONE":         0.0,
}


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Smooth sigmoid mapping to [0,1]."""
    return 1.0 / (1.0 + np.exp(-k * x))


def score_priority(
    esi_score: float,
    hz_class: str,
    snr: Optional[float],
    tmag: Optional[float],
) -> float:
    """
    Compute a combined observation priority score in [0, 1].

    Args:
        esi_score:  Earth Similarity Index [0–1]
        hz_class:   "CONSERVATIVE" | "INNER" | "OUTER" | "NONE"
        snr:        Transit SNR (TLS SDE or photometric)
        tmag:       TESS magnitude (lower = brighter = easier to follow up)

    Returns:
        priority_score in [0.0, 1.0]
    """
    esi = float(np.clip(esi_score, 0.0, 1.0))
    hz_bonus = HZ_BONUS.get(hz_class, 0.0)

    # SNR component: sigmoid centered at 10, range [7, 50]
    snr_val = float(snr) if snr is not None else 5.0
    snr_score = float(_sigmoid((snr_val - 10.0) / 5.0))

    # Brightness component: brighter is better (Tmag 6–14 mapped to [1,0])
    if tmag is not None:
        brightness = float(np.clip(1.0 - (float(tmag) - 6.0) / 10.0, 0.0, 1.0))
    else:
        brightness = 0.5

    priority = (
        0.40 * esi
        + 0.25 * hz_bonus
        + 0.20 * snr_score
        + 0.15 * brightness
    )
    return float(np.clip(priority, 0.0, 1.0))


def assign_tier(
    predicted_class: str,
    esi_score: float,
    hz_class: str,
    snr: Optional[float],
    confidence: float,
) -> int:
    """
    Assign observation tier (1–3) based on scientific priority.

    Tier 1: TRANSIT, ESI > 0.8, CONSERVATIVE HZ, SNR > 10, confidence > 0.9
    Tier 2: TRANSIT, ESI > 0.5, any HZ (not NONE), SNR > 7, confidence > 0.7
    Tier 3: All other TRANSIT candidates

    Non-TRANSIT signals get tier 0 (not applicable).
    """
    if predicted_class != "TRANSIT":
        return 0

    snr_val = float(snr) if snr is not None else 0.0

    if (esi_score >= 0.8
            and hz_class in ("CONSERVATIVE", "INNER", "OUTER")
            and snr_val >= 10.0
            and confidence >= 0.85):
        return 1
    elif (esi_score >= 0.5
            and hz_class in ("CONSERVATIVE", "INNER", "OUTER")
            and snr_val >= 7.0
            and confidence >= 0.7):
        return 2
    else:
        return 3
