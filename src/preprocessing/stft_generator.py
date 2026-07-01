"""
STFT Spectrogram Generator.
Produces 2D time-frequency spectrogram from detrended TESS flux.
NOT used as a direct model input (ECLIPSE-PRIME uses 1D views).
Used for: XAI visualization, attention heatmap overlay background.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import stft, spectrogram


def compute_stft_spectrogram(
    flux: np.ndarray,
    cadence_days: float = 2.0 / 1440.0,   # 2-min cadence in days
    nperseg: int = 256,
    noverlap: int = 192,
    nfft: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram of flux.

    Args:
        flux:          Detrended, normalized flux array
        cadence_days:  Sampling interval in days (2-min = 2/1440)
        nperseg:       Length of each STFT segment (default 256 cadences)
        noverlap:      Overlap between segments (default 192 = 75%)
        nfft:          FFT size (default 512)

    Returns:
        frequencies:   np.ndarray (F,) — frequency axis (cycles/day)
        times:         np.ndarray (T,) — time axis (cadence index)
        power_db:      np.ndarray (F, T) — power spectral density in dB
    """
    fs = 1.0 / cadence_days    # sampling frequency in cycles/day

    f, t, Zxx = stft(
        flux.astype(np.float64),
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )

    power = np.abs(Zxx) ** 2
    # Convert to dB, floor at -80 dB
    power_db = 10.0 * np.log10(power + 1e-10)
    power_db = np.clip(power_db, -80.0, None)

    return f.astype(np.float32), t.astype(np.float32), power_db.astype(np.float32)


def compute_lombscargle_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.1,
    max_period: float = 27.0,
    n_freqs: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lomb-Scargle periodogram for unevenly spaced data.
    Alternative to STFT for gap-rich light curves.

    Returns:
        periods: np.ndarray (n_freqs,) — period axis (days)
        power:   np.ndarray (n_freqs,) — normalized Lomb-Scargle power
    """
    from astropy.timeseries import LombScargle

    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    frequencies = np.linspace(freq_min, freq_max, n_freqs)
    periods = 1.0 / frequencies

    ls = LombScargle(time, flux - np.median(flux))
    power = ls.power(frequencies)

    return periods.astype(np.float32), power.astype(np.float32)


def spectrogram_to_rgb(power_db: np.ndarray) -> np.ndarray:
    """
    Convert a power spectrogram to an RGB image array for visualization.
    Uses a viridis-like colormap approximation.

    Returns: np.uint8 (H, W, 3) array.
    """
    import matplotlib.cm as cm

    # Normalize to [0, 1]
    vmin, vmax = power_db.min(), power_db.max()
    if vmax > vmin:
        normalized = (power_db - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(power_db)

    # Apply viridis colormap
    colored = cm.viridis(normalized)[:, :, :3]  # (H, W, 3) float in [0,1]
    return (colored * 255).astype(np.uint8)
