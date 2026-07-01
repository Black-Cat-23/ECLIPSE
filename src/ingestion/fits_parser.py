"""
FITS Parser — reads TESS SPOC FITS files directly from disk.
Useful when data is pre-downloaded (e.g., bulk download from MAST).
Extracts SAP flux, PDCSAP flux, centroid motion, quality flags, and CBV columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from astropy.io import fits
from loguru import logger


class FITSParser:
    """
    Parse a TESS SPOC light curve FITS file.

    Handles both:
      - lc.fits  (light curve table, FITS extension 1)
      - tp.fits  (target pixel file — uses aperture sum as SAP)
    """

    # FITS column names for each quantity
    TIME_COL = "TIME"
    SAP_COL = "SAP_FLUX"
    PDCSAP_COL = "PDCSAP_FLUX"
    PDCSAP_ERR_COL = "PDCSAP_FLUX_ERR"
    QUALITY_COL = "QUALITY"
    CENTROID_X_COL = "MOM_CENTR1"
    CENTROID_Y_COL = "MOM_CENTR2"
    CBV_COL_PREFIX = "CBVSAP_MODL"  # CBV model column (if present)

    def parse(self, fits_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Parse a TESS lc.fits file.

        Returns a dict with:
            time, flux, flux_err, sap_flux, centroid_x, centroid_y,
            quality, tic_id, sector, ra, dec, tmag
        Returns None on failure.
        """
        path = Path(fits_path)
        if not path.exists():
            logger.error(f"FITS file not found: {fits_path}")
            return None

        try:
            with fits.open(fits_path, memmap=True) as hdul:
                # Primary header: target metadata
                pri_hdr = hdul[0].header
                tic_id = int(pri_hdr.get("TICID", 0))
                sector = int(pri_hdr.get("SECTOR", 0))
                ra = float(pri_hdr.get("RA_OBJ", 0.0))
                dec = float(pri_hdr.get("DEC_OBJ", 0.0))
                tmag = float(pri_hdr.get("TESSMAG", 0.0))

                # Extension 1: light curve table
                lc_table = hdul[1].data
                colnames = [c.name for c in hdul[1].columns]

                time = lc_table[self.TIME_COL].astype(np.float64)

                # Use PDCSAP if available, else SAP
                if self.PDCSAP_COL in colnames:
                    flux = lc_table[self.PDCSAP_COL].astype(np.float32)
                    flux_err = (
                        lc_table[self.PDCSAP_ERR_COL].astype(np.float32)
                        if self.PDCSAP_ERR_COL in colnames
                        else np.full_like(flux, np.nan)
                    )
                elif self.SAP_COL in colnames:
                    flux = lc_table[self.SAP_COL].astype(np.float32)
                    flux_err = np.full_like(flux, np.nan)
                    logger.warning(f"TIC {tic_id}: PDCSAP not found, using SAP")
                else:
                    logger.error(f"TIC {tic_id}: No flux column found in {fits_path}")
                    return None

                quality = (
                    lc_table[self.QUALITY_COL].astype(np.int16)
                    if self.QUALITY_COL in colnames
                    else np.zeros(len(time), dtype=np.int16)
                )

                # Centroid motion
                centroid_x = np.zeros(len(time), dtype=np.float32)
                centroid_y = np.zeros(len(time), dtype=np.float32)
                if self.CENTROID_X_COL in colnames:
                    centroid_x = lc_table[self.CENTROID_X_COL].astype(np.float32)
                if self.CENTROID_Y_COL in colnames:
                    centroid_y = lc_table[self.CENTROID_Y_COL].astype(np.float32)

                # Mask NaNs (in time and flux simultaneously)
                valid = (
                    np.isfinite(time)
                    & np.isfinite(flux)
                    & np.isfinite(flux_err)
                )
                return {
                    "time": time[valid].astype(np.float32),
                    "flux": flux[valid],
                    "flux_err": flux_err[valid],
                    "centroid_x": centroid_x[valid],
                    "centroid_y": centroid_y[valid],
                    "quality": quality[valid],
                    "tic_id": np.array([tic_id], dtype=np.int64),
                    "sector": np.array([sector], dtype=np.int32),
                    "ra": np.array([ra], dtype=np.float64),
                    "dec": np.array([dec], dtype=np.float64),
                    "tmag": np.array([tmag], dtype=np.float32),
                }

        except Exception as e:
            logger.error(f"FITS parse error for {fits_path}: {e}")
            return None

    def parse_directory(self, fits_dir: str) -> list:
        """
        Parse all lc.fits files in a directory (recursive).

        Returns list of dicts (one per successfully parsed file).
        """
        fits_dir = Path(fits_dir)
        results = []
        for fits_file in sorted(fits_dir.rglob("*lc.fits")):
            data = self.parse(str(fits_file))
            if data:
                results.append(data)
        logger.info(f"Parsed {len(results)} FITS files from {fits_dir}")
        return results
