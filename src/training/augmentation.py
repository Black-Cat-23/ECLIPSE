"""
Augmentation utilities for ECLIPSE training.
Provides both per-batch and dataset-level augmentation strategies.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def gaussian_noise(flux: np.ndarray, sigma: float = 0.001) -> np.ndarray:
    """Add Gaussian noise to simulate photon noise variation."""
    return flux + np.random.randn(*flux.shape).astype(np.float32) * sigma


def random_phase_shift(view: np.ndarray, max_shift_frac: float = 0.05) -> np.ndarray:
    """Circular shift of a phase-folded view by up to max_shift_frac of its length."""
    max_shift = max(1, int(max_shift_frac * len(view)))
    shift = random.randint(-max_shift, max_shift)
    return np.roll(view, shift)


def flux_scaling(flux: np.ndarray, max_frac: float = 0.05) -> np.ndarray:
    """Scale flux by ±max_frac to simulate contamination/dilution."""
    scale = 1.0 + random.uniform(-max_frac, max_frac)
    return flux * scale


def transit_flip(global_view: np.ndarray) -> np.ndarray:
    """
    Randomly flip the global view left-right (phase reflection).
    Phase folded views are symmetric under reflection for circular orbits.
    """
    if random.random() < 0.5:
        return global_view[::-1].copy()
    return global_view


def augment_sample(
    raw_flux: np.ndarray,
    global_view: np.ndarray,
    local_view: np.ndarray,
    noise_sigma: float = 0.001,
    phase_shift_frac: float = 0.05,
    scale_frac: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply all augmentations to a single sample."""
    raw_flux = gaussian_noise(raw_flux, sigma=noise_sigma)
    raw_flux = flux_scaling(raw_flux, max_frac=scale_frac)
    global_view = random_phase_shift(global_view, max_shift_frac=phase_shift_frac)
    global_view = flux_scaling(global_view, max_frac=scale_frac)
    global_view = transit_flip(global_view)
    local_view = random_phase_shift(local_view, max_shift_frac=phase_shift_frac)
    local_view = flux_scaling(local_view, max_frac=scale_frac)
    return raw_flux, global_view, local_view
