"""
Sector-wide batch processor. Processes all TIC IDs in a sector using
the ECLIPSE inference pipeline, streams results to SQLite.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.inference.pipeline import ECLIPSEInferencePipeline
from src.utils.config import ECLIPSEConfig, DEFAULT_CONFIG
from src.utils.db import get_engine, get_session, init_db, upsert_candidate


class SectorBatchProcessor:
    """
    Process all light curves in a TESS sector through ECLIPSE-PRIME.
    Results are streamed to SQLite and optionally to a CSV.
    """

    def __init__(
        self,
        sector: int,
        model_path: Optional[str] = None,
        config: Optional[ECLIPSEConfig] = None,
        db_url: Optional[str] = None
    ):
        self.sector = sector
        self.config = config or DEFAULT_CONFIG
        self.pipeline = ECLIPSEInferencePipeline(
            sector=sector,
            model_path=model_path,
            config=self.config
        )
        db_url = db_url or self.config.api.db_url
        self.engine = get_engine(db_url)
        init_db(self.engine)

    def process_sector(
        self,
        tic_ids: List[int],
        progress_callback: Optional[Callable] = None,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run ECLIPSE on all TIC IDs. Saves to DB and optionally CSV.

        Args:
            tic_ids:           List of TIC IDs to process
            progress_callback: Called with (processed, total) for progress tracking
            output_csv:        If set, save all results to this CSV path

        Returns:
            DataFrame of all candidate records
        """
        results = []
        session = get_session(self.engine)
        total = len(tic_ids)

        logger.info(f"Batch processing sector {self.sector}: {total} TIC IDs")

        for i, tic_id in enumerate(tqdm(tic_ids, desc=f"Sector {self.sector}")):
            result = self.pipeline.run(tic_id)

            if result.get("error") is None and "predicted_class" in result:
                # Save to database
                try:
                    candidate_data = {
                        "tic_id": tic_id,
                        "sector": self.sector,
                        "predicted_class": result["predicted_class"],
                        "prob_transit": result["class_probs"]["TRANSIT"],
                        "prob_eb": result["class_probs"]["EB"],
                        "prob_blend": result["class_probs"]["BLEND"],
                        "prob_other": result["class_probs"]["OTHER"],
                        "confidence": result["confidence"],
                        "period": result.get("period"),
                        "period_err": result.get("period_err"),
                        "duration": result.get("duration_days"),
                        "duration_err": result.get("duration_err"),
                        "depth": result.get("depth"),
                        "depth_err": result.get("depth_err"),
                        "snr_tls": result.get("snr_tls"),
                        "snr_transit": result.get("snr_photometric"),
                        "n_transits": result.get("n_transits"),
                        "odd_even_mismatch": result.get("odd_even_mismatch"),
                    }
                    upsert_candidate(session, candidate_data)
                except Exception as e:
                    logger.warning(f"DB insert failed for TIC {tic_id}: {e}")

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        session.close()

        df = pd.DataFrame(results)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")

        logger.info(f"Batch complete. Processed {len(df)} TIC IDs in sector {self.sector}")
        return df
