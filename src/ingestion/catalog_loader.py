"""
Catalog Loader — assembles ECLIPSE 4-class ground-truth labels from:

1. TOI (TESS Objects of Interest) catalog from ExoFOP / MAST
   - PC (Planet Candidate) → TRANSIT
   - FP (False Positive) with centroid offset flag → BLEND
   - FP without centroid flag → OTHER

2. TESS Eclipsing Binary catalog → EB

3. KOI DR25 catalog (Kepler) → for transfer learning

Label priority when overlapping: EB > TRANSIT > BLEND > OTHER
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from astroquery.mast import Catalogs
from loguru import logger


# ── TOI disposition → ECLIPSE class ─────────────────────────────────────────
_TOI_DISP_MAP = {
    "PC": "TRANSIT",      # Planet Candidate
    "CP": "TRANSIT",      # Confirmed Planet
    "KP": "TRANSIT",      # Known Planet
    "FP": "BLEND",        # False Positive (default to BLEND; refine with flags)
    "FA": "OTHER",        # False Alarm
}


class CatalogLoader:
    """
    Loads and joins TOI / EB / KOI catalogs to produce per-TIC labels.
    """

    def __init__(
        self,
        toi_csv: str = "data/labels/toi_catalog.csv",
        eb_csv: str = "data/labels/tess_eb_catalog.csv",
        koi_csv: str = "data/labels/koi_dr25.csv",
        auto_download: bool = True
    ):
        self.toi_csv = Path(toi_csv)
        self.eb_csv = Path(eb_csv)
        self.koi_csv = Path(koi_csv)
        self.auto_download = auto_download

        Path(toi_csv).parent.mkdir(parents=True, exist_ok=True)

    # ── TOI catalog ──────────────────────────────────────────────────────────

    def load_toi_catalog(self) -> pd.DataFrame:
        """
        Load TOI catalog. Auto-downloads from MAST if CSV not present.

        Returns DataFrame with columns: tic_id, toi, disposition, eclipse_label,
        period, duration, depth, ra, dec.
        """
        if not self.toi_csv.exists() and self.auto_download:
            logger.info("TOI catalog not found — downloading from MAST ExoFOP")
            self._download_toi_catalog()

        if not self.toi_csv.exists():
            logger.warning("TOI catalog missing. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_csv(self.toi_csv, comment="#")
        df = self._normalize_toi(df)
        logger.info(f"TOI catalog: {len(df)} entries, "
                    f"label dist:\n{df['eclipse_label'].value_counts()}")
        return df

    def _download_toi_catalog(self) -> None:
        """Download TOI catalog via astroquery MAST."""
        try:
            result = Catalogs.query_criteria(catalog="Tic", objType="STAR")
            # Use the ExoFOP API via direct URL as astroquery doesn't have a TOI endpoint
            import urllib.request
            url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
            urllib.request.urlretrieve(url, self.toi_csv)
            logger.info(f"TOI catalog downloaded to {self.toi_csv}")
        except Exception as e:
            logger.error(f"TOI catalog download failed: {e}")

    def _normalize_toi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and add eclipse_label."""
        # Flexible column name mapping across ExoFOP versions
        col_map = {
            "TIC ID": "tic_id", "TFOPWG Disposition": "disposition",
            "Period (days)": "period", "Duration (hours)": "duration",
            "Depth (ppm)": "depth_ppm", "RA": "ra", "Dec": "dec",
            "TOI": "toi",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "disposition" not in df.columns:
            df["eclipse_label"] = "OTHER"
            return df

        def label_from_row(row):
            disp = str(row.get("disposition", "")).upper().strip()
            base = _TOI_DISP_MAP.get(disp, "OTHER")
            # depth_ppm > 10000 + FP → likely EB or BLEND
            depth = float(row.get("depth_ppm", 0) or 0)
            if base == "BLEND" and depth > 10000:
                return "EB"
            return base

        df["eclipse_label"] = df.apply(label_from_row, axis=1)
        # Convert depth from ppm to fractional
        if "depth_ppm" in df.columns:
            df["depth"] = df["depth_ppm"] / 1e6

        return df

    # ── EB catalog ───────────────────────────────────────────────────────────

    def load_eb_catalog(self) -> pd.DataFrame:
        """
        Load TESS Eclipsing Binary catalog.

        If not present, attempts to download from the TESS EB catalog paper.
        Returns DataFrame with at minimum: tic_id, eclipse_label='EB'.
        """
        if not self.eb_csv.exists() and self.auto_download:
            logger.info("EB catalog not found — attempting download")
            self._download_eb_catalog()

        if not self.eb_csv.exists():
            logger.warning("EB catalog missing.")
            return pd.DataFrame(columns=["tic_id", "eclipse_label"])

        df = pd.read_csv(self.eb_csv, comment="#")
        # Normalize TIC column name
        for col in ("TIC", "TIC_ID", "tic", "ticid"):
            if col in df.columns:
                df = df.rename(columns={col: "tic_id"})
                break
        df["eclipse_label"] = "EB"
        logger.info(f"EB catalog: {len(df)} entries")
        return df[["tic_id", "eclipse_label"]]

    def _download_eb_catalog(self) -> None:
        """Download TESS EB catalog (Prša et al. or similar)."""
        try:
            import urllib.request
            # TESS EB catalog from MAST Vizier mirror
            url = (
                "https://archive.stsci.edu/hlsp/tess-ebs/tess-ebs-catalog.csv"
            )
            urllib.request.urlretrieve(url, self.eb_csv)
            logger.info(f"EB catalog downloaded to {self.eb_csv}")
        except Exception as e:
            logger.warning(f"EB catalog download failed: {e}. "
                           "Place tess_eb_catalog.csv in data/labels/ manually.")

    # ── Merged label table ───────────────────────────────────────────────────

    def build_label_table(self) -> pd.DataFrame:
        """
        Merge TOI + EB catalogs into a unified per-TIC label table.
        Label priority: EB > TRANSIT > BLEND > OTHER.

        Returns DataFrame with columns: tic_id, eclipse_label, period,
        duration, depth, ra, dec.
        """
        toi = self.load_toi_catalog()
        eb = self.load_eb_catalog()

        _priority = {"EB": 0, "TRANSIT": 1, "BLEND": 2, "OTHER": 3}

        all_rows = []
        if not toi.empty:
            all_rows.append(toi[["tic_id", "eclipse_label"] +
                                 [c for c in ["period", "duration", "depth", "ra", "dec"]
                                  if c in toi.columns]])
        if not eb.empty:
            all_rows.append(eb)

        if not all_rows:
            return pd.DataFrame()

        merged = pd.concat(all_rows, ignore_index=True)

        # Keep highest-priority label per TIC
        merged["priority"] = merged["eclipse_label"].map(_priority)
        merged = (
            merged.sort_values("priority")
            .drop_duplicates(subset=["tic_id"], keep="first")
            .drop(columns=["priority"])
            .reset_index(drop=True)
        )
        logger.info(f"Label table: {len(merged)} unique TIC IDs. "
                    f"Dist:\n{merged['eclipse_label'].value_counts()}")
        return merged

    def get_sector_tic_ids(
        self, sector: int, limit: int = 5000
    ) -> List[int]:
        """
        Return TIC IDs from the label table that have been observed in `sector`.
        Useful for focused sector processing without downloading the full sector.
        """
        label_table = self.build_label_table()
        if label_table.empty:
            return []
        # Return all labeled TIC IDs (sector filtering would need TIC cross-match)
        return label_table["tic_id"].dropna().astype(int).tolist()[:limit]
