"""
Transit Injection-Recovery Module.
Injects synthetic batman transit signals into real TESS light curves.

Uses: (1) data augmentation for rare TRANSIT class during training
      (2) injection-recovery tests to characterize pipeline sensitivity

The batman model provides physically realistic transit shapes with
quadratic limb darkening.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class InjectedTransit:
    """Parameters of an injected synthetic transit."""
    period: float       # days
    t0: float           # first transit center (BJD)
    rp_rs: float        # planet-to-star radius ratio
    a_rs: float         # semi-major axis / stellar radius (Kepler's 3rd law)
    inc: float          # inclination (degrees)
    depth: float        # fractional transit depth = rp_rs²
    duration: float     # transit duration (days)
    u1: float = 0.4804  # quadratic LD coefficient 1
    u2: float = 0.1867  # quadratic LD coefficient 2


def inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: Optional[float] = None,
    rp_rs: Optional[float] = None,
    t0: Optional[float] = None,
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0,
    inc: float = 90.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, InjectedTransit]:
    """
    Inject a synthetic transit into a light curve.

    Parameters are randomly sampled if not provided:
      - period: Uniform[0.5, 25] days
      - rp_rs:  Log-uniform[0.01, 0.15] (sub-Earth to Jupiter-like)
      - t0:     Uniform over first period after observation start

    Args:
        time:           Time array (days)
        flux:           Clean, normalized flux (in-place modification)
        period:         Orbital period (days) — random if None
        rp_rs:          Planet-to-star radius ratio — random if None
        t0:             First transit center — random if None
        stellar_mass:   Solar masses (for Kepler's 3rd law)
        stellar_radius: Solar radii
        inc:            Orbital inclination (degrees, 90=central transit)
        seed:           Random seed for reproducibility

    Returns:
        (flux_injected, InjectedTransit params)
    """
    import batman

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    t_baseline = float(time.max() - time.min())

    # ── Sample parameters ────────────────────────────────────────────────────
    if period is None:
        period = random.uniform(0.5, min(25.0, t_baseline * 0.8))
    if rp_rs is None:
        # Log-uniform sampling: emphasizes small planets
        rp_rs = float(np.exp(random.uniform(np.log(0.01), np.log(0.15))))
    if t0 is None:
        t0 = float(time.min()) + random.uniform(0, period)

    # ── Kepler's 3rd law: a/Rs ───────────────────────────────────────────────
    # a/Rs = (G M★ / 4π²)^(1/3) × P^(2/3) / R★
    # In solar units: a/Rs ≈ (period/365.25)^(2/3) × 215 / R★
    a_rs = max(((period / 365.25) ** (2.0 / 3.0)) * 215.0 / stellar_radius, 2.0)

    # ── batman transit model ─────────────────────────────────────────────────
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.4804, 0.1867]
    params.limb_dark = "quadratic"

    try:
        m = batman.TransitModel(params, time.astype(np.float64))
        transit_model = m.light_curve(params).astype(np.float32)
    except Exception as e:
        logger.warning(f"batman model failed during injection: {e}")
        return flux.copy(), InjectedTransit(period=period, t0=t0, rp_rs=rp_rs,
                                            a_rs=a_rs, inc=inc, depth=rp_rs**2,
                                            duration=0.0)

    # Multiply (batman outputs flux normalized to 1.0)
    flux_injected = flux * transit_model

    # ── Compute transit duration (approximate) ────────────────────────────────
    # T14 = P/π × arcsin(Rs/a × sqrt((1+rp/rs)² - b²))
    b = a_rs * np.cos(np.radians(inc))  # impact parameter
    arg = np.sqrt(max((1 + rp_rs) ** 2 - b ** 2, 0)) / a_rs
    duration_days = (period / np.pi) * np.arcsin(min(arg, 1.0))

    injected = InjectedTransit(
        period=period,
        t0=t0,
        rp_rs=rp_rs,
        a_rs=a_rs,
        inc=inc,
        depth=rp_rs ** 2,
        duration=duration_days
    )

    return flux_injected, injected


def generate_injection_batch(
    time: np.ndarray,
    flux: np.ndarray,
    n_injections: int = 100,
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0
) -> list:
    """
    Generate N injection-recovery test cases from one light curve.
    Returns list of dicts with injected flux + true parameters.
    """
    results = []
    for i in range(n_injections):
        flux_inj, params = inject_transit(
            time, flux,
            stellar_mass=stellar_mass,
            stellar_radius=stellar_radius,
            seed=i
        )
        results.append({
            "time": time,
            "flux_injected": flux_inj,
            "true_period": params.period,
            "true_t0": params.t0,
            "true_depth": params.depth,
            "true_duration": params.duration,
            "true_rp_rs": params.rp_rs,
        })
    return results
