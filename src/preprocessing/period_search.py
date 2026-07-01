"""
BLS + TLS period search pipeline.

Uses Transit Least Squares (TLS) — superior to BLS because it models
realistic transit shapes (quadratic limb darkening) instead of a box function.
This improves recovery rate for grazing transits and small planets.

Iterative harmonic removal: after finding period P1, mask those transits
and search for additional periods (P2, P3) in the residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class TCE:
    """
    Threshold Crossing Event — output of period search.
    Represents one periodic dip signal found in a light curve.
    """
    tic_id: int = 0
    period: float = 0.0            # days
    t0: float = 0.0                # BJD of first transit center
    duration: float = 0.0         # hours (multiply by 24 from days)
    duration_days: float = 0.0    # days
    depth: float = 0.0            # fractional (0 to 1)
    snr: float = 0.0              # TLS Signal Detection Efficiency (SDE)
    odd_even_mismatch: float = 0.0  # |odd_depth - even_depth| / avg_depth
    n_transits: int = 0
    rp_rs: float = 0.0            # planet-to-star radius ratio = sqrt(depth)
    in_transit_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    transit_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tic_id": self.tic_id,
            "period": self.period,
            "t0": self.t0,
            "duration_hours": self.duration,
            "duration_days": self.duration_days,
            "depth": self.depth,
            "snr": self.snr,
            "odd_even_mismatch": self.odd_even_mismatch,
            "n_transits": self.n_transits,
            "rp_rs": self.rp_rs,
        }


def run_tls_search(
    time: np.ndarray,
    flux: np.ndarray,
    stellar_mass: float = 1.0,    # solar masses
    stellar_radius: float = 1.0,  # solar radii
    period_min: float = 0.5,      # days
    period_max: Optional[float] = None,
    n_top: int = 3,
    sde_threshold: float = 7.0,
    tic_id: int = 0
) -> List[TCE]:
    """
    Run TLS (Transit Least Squares) period search on a detrended light curve.

    TLS uses a realistic transit model (quadratic limb darkening) which gives
    20-30% better recovery rate than BLS for Earth-sized planets.

    Args:
        time:           Clean, detrended time array (days)
        flux:           Clean, normalized flux (median ≈ 1.0)
        stellar_mass:   Host star mass in solar masses (from TIC)
        stellar_radius: Host star radius in solar radii (from TIC)
        period_min:     Minimum search period in days
        period_max:     Maximum search period (auto = half observation baseline)
        n_top:          Number of TCEs to extract (iterative removal)
        sde_threshold:  Minimum SDE to accept a detection
        tic_id:         TIC ID (for logging and TCE metadata)

    Returns:
        List of TCE objects, sorted by SDE descending.
    """
    try:
        from transitleastsquares import transitleastsquares, cleaned_array
    except ImportError:
        logger.error("transitleastsquares not installed. Run: pip install transitleastsquares")
        return []

    if period_max is None:
        period_max = (time.max() - time.min()) / 2.0
    period_max = min(period_max, 26.0)  # Cap at 26 days for TESS 27-day sectors

    # Clean NaNs and infinite values
    try:
        time_c, flux_c = cleaned_array(time.astype(np.float64), flux.astype(np.float64))
    except Exception as e:
        logger.warning(f"TIC {tic_id}: TLS cleaned_array failed: {e}")
        return []

    if len(time_c) < 200:
        logger.debug(f"TIC {tic_id}: too few cadences for TLS ({len(time_c)})")
        return []

    results = []
    flux_work = flux_c.copy()

    for iteration in range(n_top):
        try:
            model = transitleastsquares(time_c, flux_work)
            result = model.power(
                period_min=period_min,
                period_max=period_max,
                show_progress_bar=False,
                u=[0.4804, 0.1867],   # Quadratic limb darkening (solar-like)
                limb_dark="quadratic",
                M_star=stellar_mass,
                R_star=stellar_radius,
                M_star_min=0.1,
                M_star_max=max(stellar_mass * 1.5, 5.0),
                R_star_min=0.1,
                R_star_max=max(stellar_radius * 1.5, 5.0),
                oversampling_factor=5,
                duration_grid_step=1.02
            )

            sde = float(result.SDE) if np.isfinite(result.SDE) else 0.0
            if sde < sde_threshold:
                logger.debug(f"TIC {tic_id}: SDE={sde:.2f} < {sde_threshold} at iter {iteration+1}. Stopping.")
                break

            duration_days = float(result.duration)
            tce = TCE(
                tic_id=tic_id,
                period=float(result.period),
                t0=float(result.T0),
                duration=duration_days * 24.0,      # hours
                duration_days=duration_days,
                depth=float(result.depth),
                snr=sde,
                odd_even_mismatch=float(result.odd_even_mismatch),
                n_transits=int(result.distinct_transit_count),
                rp_rs=float(result.rp_rs) if hasattr(result, "rp_rs") else np.sqrt(float(result.depth)),
                in_transit_mask=result.in_transit_mask.astype(bool),
                transit_times=list(result.transit_times) if hasattr(result, "transit_times") else []
            )
            results.append(tce)
            logger.debug(f"TIC {tic_id}: TCE {iteration+1}: P={tce.period:.4f}d, "
                         f"depth={tce.depth:.5f}, SDE={sde:.2f}, Ntransits={tce.n_transits}")

            # Mask in-transit cadences and search for next signal
            flux_work[result.in_transit_mask] = 1.0

        except Exception as e:
            logger.warning(f"TIC {tic_id}: TLS iteration {iteration+1} failed: {e}")
            break

    return results


def run_bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    n_periods: int = 10000
) -> Optional[TCE]:
    """
    BLS (Box Least Squares) fallback period search using astropy.
    Less sensitive than TLS but faster for quick pre-screening.
    """
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    if period_max is None:
        period_max = (time.max() - time.min()) / 2.0

    bls = BoxLeastSquares(time * u.day, (flux - 1.0))
    try:
        result = bls.autopower(
            0.02,  # fractional duration step
            minimum_period=period_min * u.day,
            maximum_period=period_max * u.day
        )
        best_idx = np.argmax(result.power)
        best_period = result.period[best_idx].value
        best_t0 = result.transit_time[best_idx].value
        best_duration = result.duration[best_idx].value

        # Compute depth at best period
        stats = bls.compute_stats(
            result.period[best_idx], result.duration[best_idx], result.transit_time[best_idx]
        )
        depth = float(stats["depth"][0])

        return TCE(
            period=best_period,
            t0=best_t0,
            duration=best_duration * 24.0,
            duration_days=best_duration,
            depth=abs(depth),
            snr=float(result.power[best_idx])
        )
    except Exception as e:
        logger.warning(f"BLS search failed: {e}")
        return None
