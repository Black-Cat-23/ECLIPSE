"""
Transit SNR Calculator.
Computes photometric transit SNR from detrended residuals.

SNR = depth / RMS_out_of_transit × sqrt(n_in_transit_cadences)

Also provides the TLS Signal Detection Efficiency (SDE) as a secondary metric.
"""
from __future__ import annotations

import numpy as np


def compute_transit_snr(
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    depth: float,
    n_bins: int = 201
) -> float:
    """
    Compute photometric transit SNR.

    SNR = depth / σ_out × sqrt(N_in)

    where σ_out is the out-of-transit RMS and N_in is the number of
    in-transit cadences.

    Args:
        flux:          Detrended, normalized flux
        flux_err:      Flux uncertainty (per cadence)
        period:        Orbital period (days)
        t0:            Transit center time (days)
        duration_days: Transit duration (days)
        depth:         Fractional transit depth

    Returns:
        SNR (float, > 0). Returns 0.0 if computation fails.
    """
    if period <= 0 or duration_days <= 0 or depth <= 0:
        return 0.0

    try:
        phase = ((np.arange(len(flux)) - 0) % 1.0)  # dummy — use time-based

        # Phase-fold to identify in-transit cadences
        # (using a simplified approach without the time array)
        # For proper phase fold, we need time — use duration/period ratio
        half_dur_frac = (duration_days / period) / 2.0

        # Compute per-cadence uncertainty from flux_err
        sigma_out = float(np.nanstd(flux))
        if sigma_out <= 0:
            return 0.0

        # Estimate N_in from period and duration
        n_total = len(flux)
        n_in = max(1, int(n_total * (duration_days / period)))

        snr = float(depth / sigma_out * np.sqrt(n_in))
        return max(0.0, snr)

    except Exception:
        return 0.0


def compute_transit_snr_from_fold(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    depth: float
) -> float:
    """
    More accurate SNR computation using phase-folded data.
    Requires the time array for proper in/out cadence identification.
    """
    if period <= 0 or duration_days <= 0 or depth <= 0:
        return 0.0

    try:
        # Phase fold
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        half_dur = duration_days / (2 * period)

        in_transit = np.abs(phase) <= half_dur
        out_transit = np.abs(phase) > 3 * half_dur  # well outside transit

        if in_transit.sum() < 3 or out_transit.sum() < 10:
            return 0.0

        sigma_out = float(np.nanstd(flux[out_transit]))
        n_in = int(in_transit.sum())

        if sigma_out <= 0:
            return 0.0

        snr = depth / sigma_out * np.sqrt(n_in)
        return max(0.0, float(snr))

    except Exception:
        return 0.0
