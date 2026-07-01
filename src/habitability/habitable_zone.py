"""
Habitable Zone Calculator — Kopparapu et al. (2013, 2014).

Computes the conservative and optimistic habitable zone boundaries
around a host star given its effective temperature and luminosity.

Conservative HZ:
    Inner edge: Runaway Greenhouse limit
    Outer edge: Maximum Greenhouse limit

Optimistic HZ:
    Inner edge: Recent Venus (~0.75 AU for Sun)
    Outer edge: Early Mars  (~1.77 AU for Sun)

Reference: https://arxiv.org/abs/1301.6674
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Literal

# Kopparapu 2013 Table 3 coefficients
# Format: (S_eff_sun, a, b, c, d)  for each limit
# S_eff = stellar flux relative to Earth's insolation
HZ_COEFFS = {
    "recent_venus":    (1.7763,  1.4335e-4,  3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_gh":      (1.0385,  1.2456e-4,  1.4612e-8, -7.6345e-12, -1.7511e-15),
    "moist_gh":        (1.0146,  8.1884e-5,  1.9394e-9, -4.3618e-12, -6.8260e-16),
    "max_gh":          (0.3507,  5.9578e-5,  1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars":      (0.3207,  5.4471e-5,  1.5275e-9, -2.1709e-12, -3.8282e-16),
}

T_STAR_SOLAR = 5780.0  # K


def _seff(coeffs: tuple, t_eff: float) -> float:
    """Effective stellar flux at HZ boundary for a star of T_eff (K)."""
    s0, a, b, c, d = coeffs
    t = t_eff - T_STAR_SOLAR
    return s0 + a*t + b*(t**2) + c*(t**3) + d*(t**4)


def compute_hz_bounds(
    t_eff: float,
    luminosity_lsun: float,
    conservative: bool = True,
) -> Tuple[float, float]:
    """
    Compute HZ inner and outer boundaries in AU.

    Args:
        t_eff:           Host star effective temperature (K)
        luminosity_lsun: Host star luminosity in L_sun

    Returns:
        (hz_inner_au, hz_outer_au) for conservative HZ by default.
        For optimistic: use recent_venus / early_mars limits.
    """
    t = float(np.clip(t_eff, 2600.0, 7200.0))
    l = float(max(luminosity_lsun, 1e-6))

    if conservative:
        s_inner = _seff(HZ_COEFFS["runaway_gh"], t)
        s_outer = _seff(HZ_COEFFS["max_gh"],     t)
    else:
        s_inner = _seff(HZ_COEFFS["recent_venus"], t)
        s_outer = _seff(HZ_COEFFS["early_mars"],   t)

    # a = sqrt(L / S_eff)
    hz_inner = float(np.sqrt(l / s_inner)) if s_inner > 0 else np.nan
    hz_outer = float(np.sqrt(l / s_outer)) if s_outer > 0 else np.nan
    return hz_inner, hz_outer


def compute_semi_major_axis(period_days: float, stellar_mass_msun: float) -> float:
    """
    Compute semi-major axis from Kepler's third law.
    a³ = G M P² → a (AU) = (M_star / M_sun × (P / yr)²)^(1/3)

    Returns semi-major axis in AU.
    """
    period_yr = period_days / 365.25
    a_au = (stellar_mass_msun * (period_yr ** 2)) ** (1.0 / 3.0)
    return float(a_au)


HZClass = Literal["CONSERVATIVE", "INNER", "OUTER", "NONE"]


def classify_habitable_zone(
    period_days: Optional[float],
    stellar_mass_msun: float,
    t_eff: float,
    luminosity_lsun: float,
) -> HZClass:
    """
    Classify a planet's HZ status.

    Returns:
        "CONSERVATIVE"  — inside conservative HZ
        "INNER"         — inside optimistic HZ but not conservative (inner edge)
        "OUTER"         — inside optimistic HZ but not conservative (outer edge)
        "NONE"          — outside all HZ limits
    """
    if period_days is None or period_days <= 0:
        return "NONE"

    a = compute_semi_major_axis(period_days, stellar_mass_msun)

    # Conservative limits
    con_inner, con_outer = compute_hz_bounds(t_eff, luminosity_lsun, conservative=True)
    # Optimistic limits
    opt_inner, opt_outer = compute_hz_bounds(t_eff, luminosity_lsun, conservative=False)

    if np.isnan(con_inner) or np.isnan(con_outer):
        return "NONE"

    if con_inner <= a <= con_outer:
        return "CONSERVATIVE"
    elif opt_inner <= a < con_inner:
        return "INNER"
    elif con_outer < a <= opt_outer:
        return "OUTER"
    else:
        return "NONE"
