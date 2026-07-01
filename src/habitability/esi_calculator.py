"""
Earth Similarity Index (ESI) Calculator.

The ESI is a geometric mean of four planetary similarity metrics,
each comparing a candidate planet's property to Earth's.

Formula (Schulze-Makuch et al. 2011):
    ESI = ∏ (1 - |x_i - x_earth| / (x_i + x_earth)) ^ (w_i / n)

where weights and reference Earth values are:
    - Radius:           w=0.57, x_earth=1.0 R⊕
    - Density:          w=1.07, x_earth=1.0 ρ⊕ (5514 kg/m³)
    - Escape velocity:  w=0.70, x_earth=1.0 v_esc,⊕ (11.186 km/s)
    - Surface temp:     w=5.58, x_earth=288 K (global mean)

We compute from (rp_rearth, t_eq_kelvin) using mass-radius + structure relations.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

# Earth reference values
EARTH_RADIUS_KM  = 6371.0
EARTH_DENSITY_KGM3 = 5514.0
EARTH_VESC_KMS   = 11.186
EARTH_TEMP_K     = 288.0

# ESI weights
W_RADIUS   = 0.57
W_DENSITY  = 1.07
W_VESC     = 0.70
W_TEMP     = 5.58
W_TOTAL    = W_RADIUS + W_DENSITY + W_VESC + W_TEMP


def _esi_term(x: float, x_earth: float, weight: float) -> float:
    """Single ESI similarity term, clamped to [0,1]."""
    if x <= 0 or x_earth <= 0:
        return 0.0
    ratio = abs(x - x_earth) / (x + x_earth)
    return max(0.0, min(1.0, 1.0 - ratio)) ** (weight / W_TOTAL)


def rp_to_density_kgm3(rp_rearth: float) -> float:
    """
    Estimate bulk density from radius using the Zeng et al. (2016)
    mass-radius relation:
        M/M⊕ = 4 × (R/R⊕)^3     for R < 1.5 R⊕ (rocky)
        M/M⊕ = (R/R⊕)^(1/0.55)  for R ≥ 1.5 R⊕ (sub-Neptune)
    """
    if rp_rearth < 1.5:
        mass_mearth = 4.0 * (rp_rearth ** 3.0)
    else:
        mass_mearth = rp_rearth ** (1.0 / 0.55)

    # Volume in Earth volumes = rp_rearth^3
    density_earth_units = mass_mearth / (rp_rearth ** 3)
    return density_earth_units * EARTH_DENSITY_KGM3


def rp_to_vesc_kms(rp_rearth: float) -> float:
    """
    Escape velocity from radius using same mass-radius relation.
    v_esc = √(2GM/R) ∝ √(M/R) in Earth units.
    """
    if rp_rearth < 1.5:
        mass_mearth = 4.0 * (rp_rearth ** 3.0)
    else:
        mass_mearth = rp_rearth ** (1.0 / 0.55)

    vesc_earth_units = np.sqrt(mass_mearth / rp_rearth)
    return vesc_earth_units * EARTH_VESC_KMS


def compute_esi(
    rp_rearth: Optional[float],
    t_eq_kelvin: Optional[float],
) -> float:
    """
    Compute the Earth Similarity Index.

    Args:
        rp_rearth:    Planet radius in Earth radii (from batman/TLS fit)
        t_eq_kelvin:  Equilibrium temperature in Kelvin (from stellar params)

    Returns:
        ESI in [0, 1]. Returns 0.0 if inputs are invalid.
    """
    if rp_rearth is None or t_eq_kelvin is None:
        return 0.0
    if rp_rearth <= 0 or t_eq_kelvin <= 0:
        return 0.0

    # Clamp extreme inputs
    rp = float(np.clip(rp_rearth, 0.1, 20.0))
    temp = float(np.clip(t_eq_kelvin, 50.0, 10000.0))

    density = rp_to_density_kgm3(rp)
    vesc    = rp_to_vesc_kms(rp)

    esi = (
        _esi_term(rp,      1.0,              W_RADIUS)
        * _esi_term(density, EARTH_DENSITY_KGM3, W_DENSITY)
        * _esi_term(vesc,    EARTH_VESC_KMS,     W_VESC)
        * _esi_term(temp,    EARTH_TEMP_K,       W_TEMP)
    )
    return float(np.clip(esi, 0.0, 1.0))
