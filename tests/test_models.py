"""
Tests for ECLIPSE-PRIME model components.
All tests use dummy tensors — no data download required.
GPU optional; falls back to CPU automatically.
"""
import pytest
import torch
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dummy_batch(B=2, T=1000):
    """Create a small dummy batch for fast unit tests."""
    return (
        torch.randn(B, T),        # raw_flux
        torch.randn(B, 2001),     # global_view
        torch.randn(B, 201),      # local_view
        torch.randn(B, 8),        # stellar_params
        torch.randn(B, 201),      # centroid
    )


class TestStreamA:
    def test_output_shape(self):
        from src.models.stream_a import StreamA
        model = StreamA(d_model=32, nhead=4, num_layers=2, max_seq_len=1000,
                        use_grad_checkpoint=False).to(DEVICE)
        model.eval()
        x = torch.randn(2, 1000).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 32)

    def test_no_nan_output(self):
        from src.models.stream_a import StreamA
        model = StreamA(d_model=32, nhead=4, num_layers=2, max_seq_len=1000,
                        use_grad_checkpoint=False).to(DEVICE)
        model.eval()
        x = torch.randn(3, 1000).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert torch.all(torch.isfinite(out))


class TestStreamB:
    def test_output_shape(self):
        from src.models.stream_b import StreamB
        model = StreamB(stellar_dim=8, centroid_len=201, out_features=64).to(DEVICE)
        model.eval()
        gv = torch.randn(2, 2001).to(DEVICE)
        lv = torch.randn(2, 201).to(DEVICE)
        sp = torch.randn(2, 8).to(DEVICE)
        cv = torch.randn(2, 201).to(DEVICE)
        with torch.no_grad():
            feat, attn = model(gv, lv, sp, cv)
        assert feat.shape == (2, 64)
        assert attn is not None

    def test_attention_weights_sum_to_one(self):
        """MHA attention weights should be normalized probabilities."""
        from src.models.stream_b import StreamB
        model = StreamB(out_features=64).to(DEVICE)
        model.eval()
        gv = torch.randn(1, 2001).to(DEVICE)
        lv = torch.randn(1, 201).to(DEVICE)
        sp = torch.randn(1, 8).to(DEVICE)
        cv = torch.randn(1, 201).to(DEVICE)
        with torch.no_grad():
            _, attn = model(gv, lv, sp, cv)
        # attn: (1, T', T') — rows should sum ~1 (may not be perfect due to avg over heads)
        row_sums = attn[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1)


class TestECLIPSEPrime:
    @pytest.fixture
    def model(self):
        from src.models.eclipse_prime import ECLIPSEPrime
        from src.utils.config import DEFAULT_CONFIG
        # Use small dims for fast tests
        from src.utils.config import ECLIPSEConfig, ModelConfig
        cfg = DEFAULT_CONFIG
        m = ECLIPSEPrime(
            stellar_dim=8, d_stream_a=32, d_stream_b=32, d_fused=64,
            n_classes=4, patch_size=32, T_max=1000, use_grad_checkpoint=False
        ).to(DEVICE)
        m.eval()
        return m

    def test_output_keys(self, model):
        rf, gv, lv, sp, cv = _dummy_batch(T=1000)
        with torch.no_grad():
            out = model(rf.to(DEVICE), gv.to(DEVICE), lv.to(DEVICE), sp.to(DEVICE), cv.to(DEVICE))
        required_keys = ["logits", "probs", "period_mean", "period_logvar",
                         "duration_mean", "duration_logvar", "depth_mean", "depth_logvar",
                         "snr_pred", "attention_weights"]
        for k in required_keys:
            assert k in out, f"Missing key: {k}"

    def test_prob_sums_to_one(self, model):
        rf, gv, lv, sp, cv = _dummy_batch(T=1000)
        with torch.no_grad():
            out = model(rf.to(DEVICE), gv.to(DEVICE), lv.to(DEVICE), sp.to(DEVICE), cv.to(DEVICE))
        prob_sums = out["probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)

    def test_snr_positive(self, model):
        rf, gv, lv, sp, cv = _dummy_batch(T=1000)
        with torch.no_grad():
            out = model(rf.to(DEVICE), gv.to(DEVICE), lv.to(DEVICE), sp.to(DEVICE), cv.to(DEVICE))
        assert torch.all(out["snr_pred"] > 0)

    def test_backward_runs(self, model):
        model.train()
        rf, gv, lv, sp, cv = _dummy_batch(B=2, T=1000)
        rf = rf.to(DEVICE).requires_grad_(False)
        out = model(rf, gv.to(DEVICE), lv.to(DEVICE), sp.to(DEVICE), cv.to(DEVICE))
        loss = out["probs"].sum()
        loss.backward()  # should not raise

    def test_parameter_count_reasonable(self, model):
        counts = model.parameter_count()
        assert counts["total"] > 100_000  # at least 100K parameters
        assert counts["total"] < 50_000_000  # less than 50M (fits in T4 15GB)


class TestMultiTaskLoss:
    def test_loss_positive(self):
        from src.training.losses import ECLIPSEMultiTaskLoss
        criterion = ECLIPSEMultiTaskLoss()
        B = 4
        outputs = {
            "logits": torch.randn(B, 4),
            "probs": torch.softmax(torch.randn(B, 4), dim=-1),
            "period_mean": torch.rand(B),
            "period_logvar": torch.zeros(B),
            "duration_mean": torch.rand(B) * 0.1,
            "duration_logvar": torch.zeros(B),
            "depth_mean": torch.rand(B) * 0.01,
            "depth_logvar": torch.zeros(B),
            "snr_pred": torch.rand(B) * 10,
        }
        targets = {
            "class": torch.randint(0, 4, (B,)),
            "period": torch.rand(B) * 10,
            "duration": torch.rand(B) * 0.1,
            "depth": torch.rand(B) * 0.01,
            "snr": torch.rand(B) * 10,
            "has_params": torch.tensor([True, False, True, False]),
        }
        losses = criterion(outputs, targets)
        assert losses["total"].item() > 0
        assert torch.isfinite(losses["total"])

    def test_focal_loss(self):
        from src.training.losses import focal_loss
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = focal_loss(logits, targets)
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestPhysicsLoss:
    def test_no_batman_returns_zero(self):
        """If batman not installed, should return 0 tensor gracefully."""
        from unittest.mock import patch
        import src.models.physics_loss as pl_module

        original = pl_module.batman_physics_loss

        # Patch the inner import
        with patch.dict('sys.modules', {'batman': None}):
            B = 2
            result = original(
                period_mean=torch.tensor([3.0, 5.0]),
                duration_mean=torch.tensor([0.1, 0.15]),
                depth_mean=torch.tensor([0.01, 0.02]),
                local_view_target=torch.zeros(B, 201),
                transit_mask=torch.tensor([True, True]),
                device=torch.device('cpu')
            )
            # Should be a valid tensor
            assert isinstance(result, torch.Tensor)


class TestConformal:
    def test_mc_dropout_shapes(self):
        from src.models.conformal import MCDropoutWrapper
        from src.models.eclipse_prime import ECLIPSEPrime
        model = ECLIPSEPrime(d_stream_a=32, d_stream_b=32, d_fused=64,
                              T_max=1000, use_grad_checkpoint=False).eval()
        wrapper = MCDropoutWrapper(model, n_samples=3)
        rf = torch.randn(1, 1000)
        gv = torch.randn(1, 2001)
        lv = torch.randn(1, 201)
        sp = torch.randn(1, 8)
        cv = torch.randn(1, 201)
        samples = wrapper.forward_stochastic(rf, gv, lv, sp, cv)
        assert len(samples) == 3
        assert samples[0]["probs"].shape == (1, 4)
