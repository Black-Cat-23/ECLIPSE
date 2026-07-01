"""
Phase fold pipeline — produces 1D CNN input arrays for ECLIPSE-PRIME Stream B.

Global view (2001 bins): full orbit context (full phase range [-0.5, 0.5])
Local view (201 bins):   transit morphology detail (±2× transit duration)

These views are the standard inputs for AstroNet/ExoNet-style classifiers.
Median binning is used (robust to outliers).
Normalization: subtract out-of-transit median, divide by noise estimate.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    global_bins: int = 2001,
    local_bins: int = 201,
    local_duration_factor: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve and bin into global + local views.

    Args:
        time:          Time array (days)
        flux:          Detrended, normalized flux
        period:        Orbital period (days)
        t0:            Reference transit time (days)
        duration_days: Transit duration (days)
        global_bins:   Number of bins for global view (default 2001)
        local_bins:    Number of bins for local view (default 201)
        local_duration_factor: Local view spans ±factor × duration

    Returns:
        global_view: np.float32 (global_bins,) — full orbit, normalized
        local_view:  np.float32 (local_bins,)  — transit detail, normalized
    """
    if period <= 0:
        return np.zeros(global_bins, dtype=np.float32), np.zeros(local_bins, dtype=np.float32)

    # ── Phase computation ────────────────────────────────────────────────────
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0      # center transit at phase = 0
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    # ── Global view: bin over [-0.5, 0.5] ────────────────────────────────────
    global_view = _median_bin(
        phase_sorted, flux_sorted,
        n_bins=global_bins,
        phase_min=-0.5, phase_max=0.5
    )

    # ── Local view: bin over ±2 transit durations ─────────────────────────────
    half_dur_phase = (duration_days / period) * local_duration_factor
    local_view = _median_bin(
        phase_sorted, flux_sorted,
        n_bins=local_bins,
        phase_min=-half_dur_phase,
        phase_max=half_dur_phase
    )

    # ── Normalize both views ──────────────────────────────────────────────────
    global_view = _normalize_view(global_view)
    local_view = _normalize_view(local_view)

    return global_view, local_view


def _median_bin(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    phase_min: float,
    phase_max: float
) -> np.ndarray:
    """
    Bin phase-folded flux into n_bins using median statistic.
    Empty bins are filled with 1.0 (out-of-transit continuum).
    """
    bins = np.linspace(phase_min, phase_max, n_bins + 1)
    digitized = np.digitize(phase, bins)
    binned = np.ones(n_bins, dtype=np.float64)

    for i in range(1, n_bins + 1):
        mask = digitized == i
        if mask.sum() > 0:
            binned[i - 1] = np.median(flux[mask])
        # else: keep 1.0 (continuum fill)

    return binned


def _normalize_view(view: np.ndarray) -> np.ndarray:
    """
    Normalize a phase-folded view so out-of-transit ≈ 0, transit dip < 0.

    Method: subtract median, then divide by scaled MAD (robust std estimate).
    Result is dimensionless, with noise ≈ 1.0 and transit < 0.
    """
    median = np.nanmedian(view)
    view = view - median
    mad = np.nanmedian(np.abs(view))
    scale = mad * 1.4826  # scale factor: MAD → std for Gaussian
    if scale > 1e-10:
        view = view / scale
    return view.astype(np.float32)


def fold_centroid(
    time: np.ndarray,
    centroid: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    n_bins: int = 201,
    local_duration_factor: float = 2.0
) -> np.ndarray:
    """
    Phase-fold a centroid motion time series to produce a local view
    matching the light curve local view.

    Centroid shift during transit = BLEND discriminator:
    If the centroid moves during the dip, the transit source is off-center
    (background eclipsing binary).

    Returns: np.float32 (n_bins,)
    """
    if period <= 0 or len(centroid) < 10:
        return np.zeros(n_bins, dtype=np.float32)

    # Subtract median centroid position (normalize to zero offset)
    centroid_norm = centroid - np.nanmedian(centroid)

    # Fill NaN centroids with 0 (no shift assumed)
    centroid_norm = np.where(np.isfinite(centroid_norm), centroid_norm, 0.0)

    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    centroid_s = centroid_norm[sort_idx]

    half_dur = (duration_days / period) * local_duration_factor
    binned = _median_bin(phase_s, centroid_s, n_bins, -half_dur, half_dur)

    # Normalize by out-of-transit RMS
    rms = np.sqrt(np.nanmean(binned ** 2))
    if rms > 1e-10:
        binned = binned / rms

    return binned.astype(np.float32)
