"""
End-to-end pipeline tests using synthetic data.
Tests the full ECLIPSE inference pipeline without real MAST downloads.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock


def make_synthetic_lc(n=2000, period=3.0, depth=0.01, noise=0.001):
    t = np.linspace(0, 27, n, dtype=np.float32)
    flux = np.ones(n, dtype=np.float32)
    phase = (t % period) / period
    in_transit = phase < (0.1 / period)
    flux[in_transit] -= depth
    flux += np.random.randn(n).astype(np.float32) * noise
    flux_err = np.full(n, noise, dtype=np.float32)
    return t, flux, flux_err


class TestInferencePipeline:
    @pytest.fixture
    def pipeline_with_mock_data(self):
        """Create a pipeline instance with mocked TESS fetching."""
        from src.utils.config import DEFAULT_CONFIG
        from src.inference.pipeline import ECLIPSEInferencePipeline

        pipe = ECLIPSEInferencePipeline(sector=1, config=DEFAULT_CONFIG)

        # Mock the fetcher to return synthetic data
        t, f, fe = make_synthetic_lc(n=3000, period=3.0, depth=0.01)
        mock_lc = {
            "time": t, "flux": f * 1e4, "flux_err": fe * 1e4,
            "quality": np.zeros(len(t), dtype=np.int16),
            "centroid_x": np.zeros_like(t),
            "centroid_y": np.zeros_like(t),
        }
        pipe.fetcher.get_lightcurve = MagicMock(return_value=mock_lc)
        return pipe

    def test_pipeline_run_returns_dict(self, pipeline_with_mock_data):
        result = pipeline_with_mock_data.run(tic_id=261136679)
        assert isinstance(result, dict)
        assert "tic_id" in result
        assert "predicted_class" in result

    def test_pipeline_class_is_valid(self, pipeline_with_mock_data):
        result = pipeline_with_mock_data.run(tic_id=261136679)
        if result.get("error") is None:
            assert result["predicted_class"] in ["TRANSIT", "EB", "BLEND", "OTHER"]

    def test_pipeline_probs_sum_to_one(self, pipeline_with_mock_data):
        result = pipeline_with_mock_data.run(tic_id=261136679)
        if result.get("error") is None and "class_probs" in result:
            probs = result["class_probs"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01

    def test_pipeline_snr_is_positive(self, pipeline_with_mock_data):
        result = pipeline_with_mock_data.run(tic_id=261136679)
        if result.get("snr_tls") is not None:
            assert result["snr_tls"] >= 0

    def test_pipeline_handles_missing_star(self):
        from src.utils.config import DEFAULT_CONFIG
        from src.inference.pipeline import ECLIPSEInferencePipeline

        pipe = ECLIPSEInferencePipeline(sector=1, config=DEFAULT_CONFIG)
        pipe.fetcher.get_lightcurve = MagicMock(return_value=None)

        result = pipe.run(tic_id=99999999)
        assert "error" in result
        assert result["error"] is not None


class TestSNRCalculator:
    def test_photometric_snr_positive(self):
        from src.inference.snr_calculator import compute_transit_snr
        flux = np.ones(1000, dtype=np.float32)
        flux[450:550] -= 0.01  # synthetic dip
        flux += np.random.randn(1000).astype(np.float32) * 0.001
        flux_err = np.full(1000, 0.001, dtype=np.float32)
        snr = compute_transit_snr(flux, flux_err, period=3.0, t0=0.0, duration_days=0.1, depth=0.01)
        assert snr > 0

    def test_zero_depth_returns_zero_snr(self):
        from src.inference.snr_calculator import compute_transit_snr
        flux = np.ones(1000, dtype=np.float32)
        flux_err = np.full(1000, 0.001, dtype=np.float32)
        snr = compute_transit_snr(flux, flux_err, period=3.0, t0=0.0, duration_days=0.1, depth=0.0)
        assert snr == 0.0


class TestParameterFitter:
    def test_mcmc_fitter_runs(self):
        from src.inference.parameter_fitter import BatmanMCMCFitter

        lv = np.ones(201, dtype=np.float32)
        lv[85:116] -= 0.01  # synthetic transit dip

        fitter = BatmanMCMCFitter(n_steps=100, n_walkers=8, burn_in=20)
        result = fitter.fit(
            local_view=lv,
            period_init=3.0,
            duration_init=0.1,
            depth_init=0.01
        )
        # Should return estimates (may be rough with only 100 steps)
        assert "period_median" in result
        assert "duration_median" in result
        assert "depth_median" in result
        assert result["period_median"] > 0
        assert result["depth_median"] > 0


class TestBatchProcessor:
    def test_batch_processor_empty_list(self):
        from src.inference.batch_processor import SectorBatchProcessor
        from src.utils.config import DEFAULT_CONFIG
        import tempfile, os

        # Use temporary SQLite DB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            proc = SectorBatchProcessor(
                sector=1,
                config=DEFAULT_CONFIG,
                db_url=f"sqlite:///{db_path}"
            )
            proc.pipeline.run = MagicMock(return_value={"error": "no data"})
            df = proc.process_sector([])
            assert len(df) == 0
        finally:
            if 'proc' in locals() and hasattr(proc, 'engine'):
                proc.engine.dispose()
            os.unlink(db_path)
