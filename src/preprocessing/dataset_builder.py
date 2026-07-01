"""
Dataset Builder — orchestrates the full preprocessing pipeline.

For each TIC ID in a sector:
  1. Load cached light curve (TESSFetcher)
  2. Apply quality mask + denoising
  3. Run TLS period search → TCE list
  4. For each TCE: phase fold → global + local views
  5. Extract centroid feature vector
  6. Query stellar parameters from TIC
  7. Save feature tensors to disk as .npy

Outputs:
  - data/processed/light_curves/TIC{id}_S{sector}_tce{n}.npy  (per TCE)
  - data/processed/tce_catalog.csv  (metadata for all TCEs)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.ingestion.tess_fetcher import TESSFetcher
from src.preprocessing.denoising import full_denoising_pipeline, apply_quality_mask
from src.preprocessing.period_search import run_tls_search, TCE
from src.preprocessing.phase_fold import phase_fold
from src.preprocessing.centroid_extractor import extract_centroid_feature_vector
from src.preprocessing.stellar_params import get_stellar_vector
from src.utils.config import ECLIPSEConfig, DEFAULT_CONFIG


def preprocess_single_tic(
    tic_id: int,
    sector: int,
    raw_data: Optional[dict],
    config: ECLIPSEConfig,
    stellar_cache: Optional[dict] = None,
    save_dir: str = "data/processed/light_curves"
) -> List[dict]:
    """
    Full preprocessing pipeline for a single TIC ID.
    Returns a list of TCE feature dicts (one per detected TCE).
    Each dict contains paths to saved .npy tensors + metadata.
    """
    if raw_data is None:
        return []

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Quality mask ─────────────────────────────────────────────────
    try:
        time, flux, flux_err, cx, cy = apply_quality_mask(
            raw_data["time"], raw_data["flux"], raw_data["flux_err"],
            raw_data["quality"], raw_data["centroid_x"], raw_data["centroid_y"]
        )
    except KeyError:
        # Older cache files may not have centroid / quality
        time = raw_data["time"]
        flux = raw_data["flux"]
        flux_err = raw_data.get("flux_err", np.ones_like(flux) * 0.001)
        cx = np.zeros_like(time)
        cy = np.zeros_like(time)

    if len(time) < 200:
        logger.debug(f"TIC {tic_id}: too few cadences after quality mask ({len(time)})")
        return []

    # ── Step 2: Denoising ────────────────────────────────────────────────────
    time_c, flux_c, flux_err_c = full_denoising_pipeline(
        time, flux, flux_err,
        sigma=config.data.sigma_clip_sigma,
        window_length=config.data.wotan_window_length
    )

    if len(time_c) < 100:
        return []

    # ── Step 3: TLS period search ─────────────────────────────────────────────
    stellar_vec = get_stellar_vector(tic_id, cache=stellar_cache)
    # stellar_vec[2] = mass/1.0, stellar_vec[3] = radius/1.0
    stellar_mass = float(stellar_vec[2]) * 1.0   # re-scale back to solar
    stellar_radius = float(stellar_vec[3]) * 1.0

    tces = run_tls_search(
        time_c, flux_c,
        stellar_mass=stellar_mass,
        stellar_radius=stellar_radius,
        period_min=config.data.tls_period_min,
        period_max=config.data.tls_period_max,
        n_top=3,
        sde_threshold=config.data.tls_sde_threshold,
        tic_id=tic_id
    )

    # Also include "no-transit" record for OTHER class if no TCE found
    # (used in training with label=OTHER)
    if not tces:
        # Store raw flux for Stream A even without a period
        raw_flux_norm = _pad_or_truncate(flux_c, config.model.T_max)
        npy_path = save_path / f"TIC{tic_id}_S{sector:02d}_notce.npy"
        np.save(npy_path, {"raw_flux": raw_flux_norm, "stellar": stellar_vec})
        return []

    # ── Steps 4–6: Phase fold + centroid + stellar per TCE ───────────────────
    results = []
    for n, tce in enumerate(tces):
        try:
            global_view, local_view = phase_fold(
                time_c, flux_c,
                period=tce.period,
                t0=tce.t0,
                duration_days=tce.duration_days,
                global_bins=config.data.global_view_bins,
                local_bins=config.data.local_view_bins
            )

            centroid_view = extract_centroid_feature_vector(
                time_c, cx[:len(time_c)], cy[:len(time_c)],
                period=tce.period,
                t0=tce.t0,
                duration_days=tce.duration_days,
                n_bins=config.data.local_view_bins
            )

            # Pad/truncate raw flux to T_max for Stream A
            raw_flux_padded = _pad_or_truncate(flux_c, config.model.T_max)

            # ── Save tensors ─────────────────────────────────────────────────
            tensor_name = f"TIC{tic_id}_S{sector:02d}_tce{n}"
            tensor_path = save_path / f"{tensor_name}.npz"
            np.savez_compressed(
                tensor_path,
                raw_flux=raw_flux_padded,
                global_view=global_view,
                local_view=local_view,
                centroid=centroid_view,
                stellar=stellar_vec,
            )

            results.append({
                "tic_id": tic_id,
                "sector": sector,
                "tce_n": n,
                "tensor_path": str(tensor_path),
                "period": tce.period,
                "t0": tce.t0,
                "duration_hours": tce.duration,
                "duration_days": tce.duration_days,
                "depth": tce.depth,
                "snr_tls": tce.snr,
                "odd_even_mismatch": tce.odd_even_mismatch,
                "n_transits": tce.n_transits,
                "rp_rs": tce.rp_rs,
            })

        except Exception as e:
            logger.warning(f"TIC {tic_id} TCE {n}: preprocessing failed: {e}")
            continue

    return results


def build_tce_catalog(
    sector: int = 1,
    config: Optional[ECLIPSEConfig] = None,
    tic_ids: Optional[List[int]] = None,
    label_df: Optional[pd.DataFrame] = None,
    max_tic: int = 5000,
    output_csv: str = "data/processed/tce_catalog.csv"
) -> pd.DataFrame:
    """
    Orchestrate full preprocessing for an entire sector.

    Args:
        sector:     TESS sector number
        config:     ECLIPSEConfig (defaults to DEFAULT_CONFIG)
        tic_ids:    List of TIC IDs to process (auto-fetched if None)
        label_df:   DataFrame with columns [tic_id, eclipse_label, period, ...].
                    If provided, labels are joined to the output catalog.
        max_tic:    Maximum number of TIC IDs to process (for testing)
        output_csv: Path to save the TCE catalog

    Returns:
        tce_catalog: pd.DataFrame with one row per TCE
    """
    if config is None:
        config = DEFAULT_CONFIG

    fetcher = TESSFetcher(sector=sector, output_dir=config.data.raw_dir)
    stellar_cache: dict = {}

    # ── Get TIC list ─────────────────────────────────────────────────────────
    if tic_ids is None:
        if label_df is not None and not label_df.empty:
            tic_ids = label_df["tic_id"].astype(int).tolist()[:max_tic]
        else:
            tic_ids = fetcher.get_sector_tic_list(limit=max_tic)

    logger.info(f"Building TCE catalog for sector {sector}: {len(tic_ids)} TIC IDs")

    all_rows = []
    for tic_id in tqdm(tic_ids, desc=f"Sector {sector}"):
        raw_data = fetcher.get_lightcurve(tic_id)
        rows = preprocess_single_tic(
            tic_id=tic_id,
            sector=sector,
            raw_data=raw_data,
            config=config,
            stellar_cache=stellar_cache,
            save_dir=str(Path(config.data.processed_dir) / "light_curves")
        )
        all_rows.extend(rows)

    catalog = pd.DataFrame(all_rows)

    # ── Join labels ───────────────────────────────────────────────────────────
    if label_df is not None and not label_df.empty and not catalog.empty:
        catalog = catalog.merge(
            label_df[["tic_id", "eclipse_label"]].drop_duplicates("tic_id"),
            on="tic_id",
            how="left"
        )
        catalog["eclipse_label"] = catalog["eclipse_label"].fillna("OTHER")
    elif not catalog.empty:
        catalog["eclipse_label"] = "UNKNOWN"

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_csv, index=False)
    logger.info(f"TCE catalog saved: {output_csv} ({len(catalog)} TCEs)")
    if not catalog.empty and "eclipse_label" in catalog.columns:
        logger.info(f"Label distribution:\n{catalog['eclipse_label'].value_counts()}")

    return catalog


def _pad_or_truncate(arr: np.ndarray, T_max: int) -> np.ndarray:
    """Pad with zeros or truncate to exactly T_max length."""
    if len(arr) >= T_max:
        return arr[:T_max].astype(np.float32)
    padded = np.zeros(T_max, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded
