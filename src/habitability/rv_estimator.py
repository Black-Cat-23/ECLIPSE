"""
Radial Velocity Semi-Amplitude Estimator.

Estimates the expected RV signal K for a planet candidate to assess
whether ground-based RV follow-up is feasible.

Formula (circular orbit, edge-on approximation):
    K = (2π G / P)^(1/3) × M_p sin(i) / (M_star + M_p)^(2/3)

Simplified using Earth-mass from radius via mass-radius relation,
and assuming sin(i) ≈ 1 (worst case / edge-on).
"""
from __future__ import annotations
import numpy as np
from typing import Optional


# Physical constants
G_SI = 6.674e-11        # m³ kg⁻¹ s⁻²
M_SUN_KG = 1.989e30     # kg
M_EARTH_KG = 5.972e24   # kg
DAY_S = 86400.0         # seconds per day


def radius_to_mass_mearth(rp_rearth: float) -> float:
    """
    Estimate planet mass from radius using empirical mass-radius relation
    (Zeng et al. 2016 / Chen & Kipping 2017 blend):
        R < 1.5 R⊕: rocky, M ∝ R^3.7  (Zeng rocky)
        1.5–4 R⊕:  volatile-rich, M ∝ R^1.74  (Chen & Kipping transition)
        > 4 R⊕:    giant, M ∝ R^1.0   (roughly)
    """
    r = float(rp_rearth)
    if r <= 1.5:
        return 0.9718 * (r ** 3.58)
    elif r <= 4.0:
        return 2.137 * (r ** 1.74)
    else:
        return 1.4 * (r ** 1.0) * 317.8  # Jupiter regime


def estimate_rv_amplitude(
    period_days: Optional[float],
    rp_rearth: Optional[float],
    stellar_mass_msun: Optional[float] = 1.0,
    inclination_deg: float = 90.0,
) -> Optional[float]:
    """
    Estimate radial velocity semi-amplitude K in m/s.

    Args:
        period_days:       Orbital period (days)
        rp_rearth:         Planet radius in Earth radii
        stellar_mass_msun: Host star mass in solar masses
        inclination_deg:   Orbital inclination (default 90° = edge-on)

    Returns:
        K in m/s, or None if inputs are invalid.
    """
    if period_days is None or rp_rearth is None:
        return None
    if period_days <= 0 or rp_rearth <= 0:
        return None

    # Planet mass in kg
    m_planet_mearth = radius_to_mass_mearth(rp_rearth)
    m_planet_kg = m_planet_mearth * M_EARTH_KG

    # Stellar mass in kg
    m_star_kg = float(stellar_mass_msun or 1.0) * M_SUN_KG

    # Period in seconds
    p_s = period_days * DAY_S

    # K = (2π G / P)^(1/3) × m_p × sin(i) / (m_star + m_p)^(2/3)
    sin_i = float(np.sin(np.radians(inclination_deg)))

    k = (
        ((2.0 * np.pi * G_SI) / p_s) ** (1.0 / 3.0)
        * m_planet_kg * sin_i
        / ((m_star_kg + m_planet_kg) ** (2.0 / 3.0))
    )

    return float(round(k, 4))  # m/s
