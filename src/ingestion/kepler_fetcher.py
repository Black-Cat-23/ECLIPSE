"""
Kepler KOI DR25 Light Curve Fetcher.
Downloads Kepler long-cadence light curves for transfer learning pretraining.
KOI DR25 provides high-quality labeled TRANSIT / FP / CANDIDATE data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import lightkurve as lk
import numpy as np
import pandas as pd
from loguru import logger


class KeplerFetcher:
    """
    Fetches Kepler KIC light curves from MAST for transfer pretraining.

    The KOI DR25 dataset (~8000 KOIs) is the gold standard for training
    exoplanet classifiers. ECLIPSE uses it for Stream B CNN pretraining
    before fine-tuning on TESS data.
    """

    def __init__(self, output_dir: str = "data/raw/kepler", quarter: int = 9):
        """
        Args:
            output_dir: Directory to cache downloaded NPZ files.
            quarter:    Kepler quarter (1-17). Quarter 9 is commonly used
                        as the reference for KOI vetting (continuous 90-day).
        """
        self.output_dir = Path(output_dir)
        self.quarter = quarter
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_lightcurve(self, kic_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Download (or load from cache) a Kepler PDC light curve.

        Returns dict with keys: time, flux, flux_err, kic_id, quarter.
        """
        cache_path = self.output_dir / f"KIC{kic_id}_Q{self.quarter:02d}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            return {k: data[k] for k in data.files}

        try:
            search = lk.search_lightcurve(
                f"KIC {kic_id}",
                quarter=self.quarter,
                mission="Kepler",
                author="Kepler"
            )
            if len(search) == 0:
                return None

            lc = search[0].download()
            if lc is None:
                return None

            lc = lc.remove_nans()
            result = {
                "time": lc.time.value.astype(np.float32),
                "flux": lc.flux.value.astype(np.float32),
                "flux_err": lc.flux_err.value.astype(np.float32),
                "kic_id": np.array([kic_id], dtype=np.int64),
                "quarter": np.array([self.quarter], dtype=np.int32),
            }
            np.savez_compressed(cache_path, **result)
            return result

        except Exception as e:
            logger.warning(f"KIC {kic_id} Q{self.quarter}: fetch failed — {e}")
            return None

    def batch_download(
        self, kic_ids: List[int], max_workers: int = 4
    ) -> pd.DataFrame:
        """Download multiple KIC IDs in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_lightcurve, kid): kid for kid in kic_ids}
            for future in as_completed(futures):
                kid = futures[future]
                data = future.result()
                results.append({"kic_id": kid, "downloaded": data is not None})
        return pd.DataFrame(results)

    @staticmethod
    def load_koi_dr25_catalog(csv_path: str = "data/labels/koi_dr25.csv") -> pd.DataFrame:
        """
        Load the KOI DR25 cumulative table.

        Expected columns: kepid, kepoi_name, koi_disposition, koi_period,
        koi_duration, koi_depth, koi_prad.

        Maps koi_disposition → ECLIPSE 4-class labels:
            CONFIRMED → TRANSIT
            CANDIDATE → TRANSIT (uncertain)
            FALSE POSITIVE → BLEND or OTHER (based on koi_fpflag_co)
        """
        df = pd.read_csv(csv_path, comment="#")
        # Rename for convenience
        df = df.rename(columns={"kepid": "kic_id"})

        # Map to 4-class labels
        def map_label(row):
            disp = str(row.get("koi_disposition", "")).upper()
            if disp in ("CONFIRMED", "CANDIDATE"):
                return "TRANSIT"
            # Centroid offset flag → BLEND; else OTHER
            fp_co = int(row.get("koi_fpflag_co", 0))
            return "BLEND" if fp_co else "OTHER"

        df["eclipse_label"] = df.apply(map_label, axis=1)
        logger.info(f"KOI DR25: loaded {len(df)} entries. "
                    f"Label distribution:\n{df['eclipse_label'].value_counts()}")
        return df
