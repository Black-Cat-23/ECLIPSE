"""
Centroid Extractor — extracts and processes centroid motion from TESS light curves.

Centroid shift during transit is the primary discriminator for BLEND class:
  - TRANSIT: centroid stationary during dip (flux source is the target star)
  - BLEND:   centroid moves during dip (flux dip from background EB)

Produces: centroid displacement time series + phase-folded centroid view.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from loguru import logger


def compute_centroid_displacement(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    method: str = "rms"
) -> np.ndarray:
    """
    Compute scalar centroid displacement from (x, y) centroid tracks.

    Args:
        centroid_x: Column centroid time series (pixels)
        centroid_y: Row centroid time series (pixels)
        method: "rms" → sqrt(dx² + dy²); "x" → use only column centroid

    Returns:
        displacement: np.float32 (N,) — centroid displacement in pixels
    """
    # Subtract median reference position
    x_ref = np.nanmedian(centroid_x)
    y_ref = np.nanmedian(centroid_y)
    dx = centroid_x - x_ref
    dy = centroid_y - y_ref

    # Fill NaN with zero displacement
    dx = np.where(np.isfinite(dx), dx, 0.0)
    dy = np.where(np.isfinite(dy), dy, 0.0)

    if method == "rms":
        displacement = np.sqrt(dx ** 2 + dy ** 2)
    elif method == "x":
        displacement = np.abs(dx)
    elif method == "y":
        displacement = np.abs(dy)
    else:
        displacement = np.sqrt(dx ** 2 + dy ** 2)

    return displacement.astype(np.float32)


def extract_centroid_feature_vector(
    time: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    n_bins: int = 201,
    local_duration_factor: float = 2.0
) -> np.ndarray:
    """
    Extract the phase-folded centroid displacement local view.
    This is the 201-point centroid input to Stream B.

    A large centroid shift in-transit vs out-of-transit → BLEND class.

    Returns: np.float32 (n_bins,)
    """
    from src.preprocessing.phase_fold import fold_centroid

    displacement = compute_centroid_displacement(centroid_x, centroid_y)

    return fold_centroid(
        time=time,
        centroid=displacement,
        period=period,
        t0=t0,
        duration_days=duration_days,
        n_bins=n_bins,
        local_duration_factor=local_duration_factor
    )


def compute_centroid_in_out_ratio(
    time: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    period: float,
    t0: float,
    duration_days: float
) -> float:
    """
    Compute ratio of in-transit vs out-of-transit centroid RMS.
    Ratio > 1.5 is a strong BLEND indicator.

    Returns: float (centroid_in_rms / centroid_out_rms)
    """
    displacement = compute_centroid_displacement(centroid_x, centroid_y)

    if period <= 0:
        return 1.0

    # Phase-fold
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    half_dur = duration_days / (2 * period)
    in_transit = np.abs(phase) <= half_dur
    out_of_transit = np.abs(phase) > 3 * half_dur  # well outside transit

    if in_transit.sum() < 3 or out_of_transit.sum() < 10:
        return 1.0

    rms_in = float(np.sqrt(np.mean(displacement[in_transit] ** 2)))
    rms_out = float(np.sqrt(np.mean(displacement[out_of_transit] ** 2)))

    if rms_out < 1e-10:
        return 1.0

    return rms_in / rms_out
