"""
TESS Light Curve Fetcher
Downloads PDCSAP flux from MAST for a given sector using lightkurve.

Handles the actual TESS high-cadence data as specified by PS-07:
  - Uses archive.stsci.edu / MAST portal (2-min cadence SPOC)
  - Downloads and caches ~20-30k light curves per sector
  - Preserves centroid time series for BLEND detection
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import lightkurve as lk
import numpy as np
import pandas as pd
from loguru import logger


class TESSFetcher:
    """
    Fetches and caches TESS PDCSAP light curves from MAST.

    Light curves are cached as compressed NPZ files so subsequent runs
    don't re-download. Centroid columns (MOM_CENTR1/2) are preserved
    for BLEND vs TRANSIT discrimination.

    Usage:
        fetcher = TESSFetcher(sector=1, output_dir='data/raw')
        lc_data = fetcher.get_lightcurve(tic_id=261136679)
    """

    def __init__(
        self,
        sector: int,
        output_dir: str = "data/raw",
        cadence: str = "2min",
        author: str = "SPOC"
    ):
        self.sector = sector
        self.output_dir = Path(output_dir) / f"sector_{sector:02d}"
        self.cadence = cadence
        self.author = author
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_lightcurve(self, tic_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Download (or load from cache) a TESS PDCSAP light curve.

        Returns dict with keys:
            time        : np.float32 (N,) — BTJD timestamps
            flux        : np.float32 (N,) — PDCSAP normalized flux
            flux_err    : np.float32 (N,) — flux uncertainty
            centroid_x  : np.float32 (N,) — MOM_CENTR1 (column)
            centroid_y  : np.float32 (N,) — MOM_CENTR2 (row)
            quality     : np.int16   (N,) — TESS quality flags
        Returns None if data unavailable.
        """
        cache_path = self.output_dir / f"TIC{tic_id}_S{self.sector:02d}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            return {k: data[k] for k in data.files}

        try:
            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                sector=self.sector,
                author=self.author,
                exptime=120
            )
            if len(search) == 0:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    sector=self.sector,
                    author=self.author,
                    cadence="short"
                )
            if len(search) == 0:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    sector=self.sector,
                    author=self.author
                )

            if len(search) == 0:
                logger.debug(f"TIC {tic_id}: no SPOC data in sector {self.sector}")
                return None

            # Download the first result (SPOC 2-min is preferred)
            lc_collection = search[0].download()
            if lc_collection is None:
                return None

            # Select PDCSAP flux (systematics-corrected, not detrended)
            lc = lc_collection.select_flux("pdcsap_flux")
            lc = lc.remove_nans()

            raw = lc_collection  # keep original for centroid extraction
            time_arr = lc.time.value.astype(np.float32)
            flux_arr = lc.flux.value.astype(np.float32)
            flux_err = lc.flux_err.value.astype(np.float32)
            quality = lc.quality.value.astype(np.int16)

            # Extract centroid motion (key for BLEND detection)
            centroid_x = np.zeros_like(flux_arr)
            centroid_y = np.zeros_like(flux_arr)
            try:
                if hasattr(raw, "centroid_col") and raw.centroid_col is not None:
                    cx = raw.centroid_col.value
                    cy = raw.centroid_row.value
                    # Align to NaN-removed time
                    valid = ~np.isnan(lc_collection.flux.value)
                    centroid_x = cx[valid].astype(np.float32)
                    centroid_y = cy[valid].astype(np.float32)
            except Exception:
                pass  # Centroid not available for all targets; zeros = no shift

            result = {
                "time": time_arr,
                "flux": flux_arr,
                "flux_err": flux_err,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "quality": quality,
                "tic_id": np.array([tic_id], dtype=np.int64),
                "sector": np.array([self.sector], dtype=np.int32),
            }
            np.savez_compressed(cache_path, **result)
            logger.debug(f"TIC {tic_id}: downloaded {len(time_arr)} cadences")
            return result

        except Exception as e:
            logger.warning(f"TIC {tic_id}: fetch failed — {e}")
            return None

    def batch_download(
        self,
        tic_ids: List[int],
        max_workers: int = 8,
        delay_between: float = 0.1
    ) -> pd.DataFrame:
        """
        Download a list of TIC IDs in parallel (ThreadPoolExecutor).

        Returns a DataFrame with columns: tic_id, downloaded, n_cadences.
        """
        results = []
        logger.info(f"Batch downloading {len(tic_ids)} TIC IDs (sector {self.sector})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_with_delay, tid, delay_between): tid
                for tid in tic_ids
            }
            for future in as_completed(futures):
                tic_id = futures[future]
                data = future.result()
                results.append({
                    "tic_id": tic_id,
                    "downloaded": data is not None,
                    "n_cadences": len(data["time"]) if data else 0
                })

        df = pd.DataFrame(results)
        n_ok = df["downloaded"].sum()
        logger.info(f"Batch download complete: {n_ok}/{len(tic_ids)} successful")
        return df

    def _download_with_delay(self, tic_id: int, delay: float) -> Optional[dict]:
        time.sleep(delay)
        return self.get_lightcurve(tic_id)

    def get_sector_tic_list(self, limit: int = 30000) -> List[int]:
        """
        Query MAST for all SPOC 2-min TIC IDs observed in this sector.

        Returns a list of integer TIC IDs ready for batch_download.
        Note: Full sector queries can take 30–60 s due to MAST latency.
        """
        logger.info(f"Querying MAST for sector {self.sector} TIC IDs (limit={limit})")
        try:
            search_results = lk.search_lightcurve(
                "*",
                sector=self.sector,
                cadence=self.cadence,
                mission="TESS",
                author=self.author,
                limit=limit
            )
            tic_ids = []
            for name in search_results.target_name:
                try:
                    # target_name format: "TIC 261136679"
                    tic_ids.append(int(str(name).strip().split()[-1]))
                except (ValueError, IndexError):
                    continue
            logger.info(f"Found {len(tic_ids)} TIC IDs in sector {self.sector}")
            return tic_ids
        except Exception as e:
            logger.error(f"Sector TIC list query failed: {e}")
            return []
