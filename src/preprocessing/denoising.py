"""
Light curve denoising pipeline.
Applies: sigma-clip → Wotan biweight detrending → normalization.

The Wotan biweight method is robust to outliers (transits are outliers!)
and preserves transit shape better than simple polynomial fits.
CBV correction is handled by PDCSAP selection in the fetcher.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from astropy.stats import sigma_clip as astropy_sigma_clip
from wotan import flatten
from loguru import logger


def full_denoising_pipeline(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    sigma: float = 3.0,
    window_length: float = 0.5,   # days — Wotan sliding window
    min_cadences: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete denoising pipeline for a single TESS PDCSAP light curve.

    Steps:
      1. Median normalize: flux / median(flux)
      2. Sigma clipping: iterative 3σ, 5 iterations (astropy)
      3. Wotan biweight detrending: removes stellar variability & long trends
         while preserving transit morphology
      4. Final normalization: median → 0.0 (flux centered)

    Args:
        time:           BTJD timestamps (days)
        flux:           PDCSAP flux (raw electron/s or normalized)
        flux_err:       Flux uncertainty
        sigma:          Sigma-clip threshold (default 3.0)
        window_length:  Wotan sliding window length in days (default 0.5)
        min_cadences:   Minimum cadences required to attempt detrending

    Returns:
        (time_clean, flux_clean, flux_err_clean) all float32
        flux_clean is normalized: out-of-transit median ≈ 1.0
    """
    if len(time) < min_cadences:
        logger.warning(f"Skipping denoising: only {len(time)} cadences")
        return time.astype(np.float32), flux.astype(np.float32), flux_err.astype(np.float32)

    # ── Step 1: Median normalize ─────────────────────────────────────────────
    median_flux = np.nanmedian(flux)
    if median_flux <= 0:
        logger.warning("Median flux <= 0; skipping normalization")
        median_flux = 1.0
    flux_norm = np.array(flux / median_flux, dtype=float, copy=True)
    flux_err_norm = np.array(flux_err / median_flux, dtype=float, copy=True)

    # ── Step 2: Sigma clipping ───────────────────────────────────────────────
    clipped = astropy_sigma_clip(flux_norm, sigma=sigma, maxiters=5, masked=True)
    mask = ~clipped.mask  # True = good cadence

    time_c = time[mask]
    flux_c = flux_norm[mask]
    flux_err_c = flux_err_norm[mask]

    if len(time_c) < min_cadences:
        logger.warning("Too many cadences sigma-clipped; using pre-clip data")
        time_c, flux_c, flux_err_c = time, flux_norm, flux_err_norm

    # ── Step 3: Wotan biweight detrending ────────────────────────────────────
    # biweight is robust to transit dips (outliers below continuum)
    # window_length=0.5 days ~ 360 cadences for 2-min TESS
    flatten_flux = flux_c.copy()
    if len(time_c) >= min_cadences:
        try:
            flatten_flux, _trend = flatten(
                time_c,
                flux_c,
                method="biweight",
                window_length=window_length,
                return_trend=True,
                robust=True,
                edge_cutoff=0.0
            )
        except Exception as e:
            logger.warning(f"Wotan detrending failed: {e}; using sigma-clipped flux")
            flatten_flux = flux_c

    # ── Step 4: Final normalization → median = 1.0 ────────────────────────────
    med = np.nanmedian(flatten_flux)
    if med > 0:
        flatten_flux = flatten_flux / med
        flux_err_final = flux_err_c / med
    else:
        flux_err_final = flux_err_c

    return (
        time_c.astype(np.float32),
        flatten_flux.astype(np.float32),
        flux_err_final.astype(np.float32)
    )


def apply_quality_mask(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    quality_bitmask: int = 175
) -> Tuple[np.ndarray, ...]:
    """
    Apply TESS quality bitmask to remove bad cadences.

    Default bitmask=175 removes: attitude tweaks, safe mode, coarse pointing,
    Earth-point, argabrightening, stray light. Keeps: cosmic rays (bit 5)
    since those are handled by sigma clipping.

    Returns masked arrays: time, flux, flux_err, centroid_x, centroid_y.
    """
    good = (quality & quality_bitmask) == 0
    return (
        time[good], flux[good], flux_err[good],
        centroid_x[good], centroid_y[good]
    )
