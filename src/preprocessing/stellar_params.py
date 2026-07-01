"""
Stellar Parameters Extractor.
Queries TESS Input Catalog (TIC) via astroquery for host star properties.
Returns an 8-dimensional feature vector used by Stream B's stellar MLP.

Features: [Teff, log_g, M★, R★, [Fe/H], distance, Tmag, contamination_ratio]
All values are normalized to roughly unit scale for the MLP.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from loguru import logger


# ── Default values (solar-like) when catalog values are missing ──────────────
STELLAR_DEFAULTS = {
    "Teff": 5778.0,     # Kelvin (Sun)
    "logg": 4.44,       # log(cm/s²)
    "mass": 1.0,        # Solar masses
    "rad": 1.0,         # Solar radii
    "MH": 0.0,          # [Fe/H] solar
    "d": 100.0,         # parsecs (rough guess)
    "Tmag": 10.0,       # TESS magnitude
    "contratio": 0.0,   # contamination ratio
}

# ── Normalization scales (for MLP input normalization) ────────────────────────
STELLAR_SCALES = {
    "Teff": 5778.0,
    "logg": 4.44,
    "mass": 1.0,
    "rad": 1.0,
    "MH": 1.0,
    "d": 1000.0,
    "Tmag": 12.0,
    "contratio": 1.0,
}


def query_tic(tic_id: int) -> Dict[str, float]:
    """
    Query the TESS Input Catalog for stellar parameters.

    Returns a dict with raw catalog values (before normalization).
    Uses astroquery.mast.Catalogs with catalog='Tic'.
    Falls back to solar defaults if query fails or values are missing.
    """
    try:
        from astroquery.mast import Catalogs
        result = Catalogs.query_criteria(catalog="Tic", ID=tic_id)

        if len(result) == 0:
            logger.debug(f"TIC {tic_id}: not found in TIC; using defaults")
            return STELLAR_DEFAULTS.copy()

        row = result[0]
        params = {}
        for key, default in STELLAR_DEFAULTS.items():
            val = row[key] if key in result.colnames else None
            if val is None or (hasattr(val, "mask") and val.mask):
                params[key] = default
            else:
                v = float(val)
                params[key] = v if np.isfinite(v) else default

        return params

    except Exception as e:
        logger.warning(f"TIC {tic_id}: TIC query failed ({e}); using defaults")
        return STELLAR_DEFAULTS.copy()


def stellar_params_to_vector(params: Dict[str, float]) -> np.ndarray:
    """
    Convert a stellar params dict to a normalized 8-dim float32 vector.

    Normalization:
        - Teff / 5778       → ~1 for solar, range [0.5, 2.0]
        - logg / 4.44       → ~1 for MS star, range [0.5, 1.5]
        - mass / 1.0        → solar masses
        - rad / 1.0         → solar radii
        - MH (Fe/H)         → direct (range -2 to +0.5)
        - d / 1000          → kpc (range 0.01 to 10+)
        - Tmag / 12         → dimensionless
        - contratio         → direct (range 0 to 1)

    Returns: np.float32 (8,)
    """
    keys = ["Teff", "logg", "mass", "rad", "MH", "d", "Tmag", "contratio"]
    vec = []
    for k in keys:
        raw = params.get(k, STELLAR_DEFAULTS[k])
        scale = STELLAR_SCALES[k]
        vec.append(float(raw) / scale if scale != 0 else float(raw))

    arr = np.array(vec, dtype=np.float32)
    # Clip to reasonable range to avoid exploding activations
    arr = np.clip(arr, -10.0, 10.0)
    return arr


def get_stellar_vector(tic_id: int, cache: Optional[Dict] = None) -> np.ndarray:
    """
    High-level function: query TIC + convert to normalized 8-dim vector.
    
    Args:
        tic_id: TESS Input Catalog ID
        cache:  Optional dict {tic_id: params_dict} to avoid repeated queries

    Returns: np.float32 (8,)
    """
    if cache is not None and tic_id in cache:
        return stellar_params_to_vector(cache[tic_id])

    params = query_tic(tic_id)

    if cache is not None:
        cache[tic_id] = params

    return stellar_params_to_vector(params)
