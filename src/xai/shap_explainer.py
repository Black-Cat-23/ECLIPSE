"""
SHAP DeepExplainer — Feature Importance for ECLIPSE-PRIME.

Uses SHAP's DeepExplainer (gradient-based) on the model's classifier head
to compute per-feature importance values for any given prediction.

Feature names correspond to the stellar parameter vector:
    [Teff, logg, mass, radius, FeH, distance, Tmag, contamination]
plus engineered transit features:
    [period, duration, depth, snr, rp_rs, odd_even_diff]
"""
from __future__ import annotations

import json
import numpy as np
from typing import List, Dict, Optional, Any
from loguru import logger

# Feature name map (matches stellar_params.py + TCE features)
STELLAR_FEATURE_NAMES = [
    "Teff (K)", "log g", "Stellar Mass (M☉)", "Stellar Radius (R☉)",
    "[Fe/H]", "Distance (pc)", "TESS Magnitude", "Contamination Ratio"
]

TRANSIT_FEATURE_NAMES = [
    "Period (days)", "Duration (hrs)", "Depth (ppm)", "TLS SDE",
    "Rp/Rs", "Odd-Even Diff."
]


class SHAPExplainer:
    """
    Wraps a PyTorch model to produce SHAP explanations.

    Uses SHAP KernelExplainer as the fallback (model-agnostic) when
    DeepExplainer is not applicable (e.g. complex multi-input architecture).
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def explain(
        self,
        stellar_vec: np.ndarray,
        tce_features: np.ndarray,
        n_background: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Compute SHAP values for the stellar + TCE feature vectors.

        Returns list of dicts:
            [{name, value, shap_value}, ...] sorted by |shap_value| descending.
        """
        try:
            return self._explain_kernel(stellar_vec, tce_features)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}. Using gradient proxy.")
            return self._gradient_proxy(stellar_vec, tce_features)

    def _explain_kernel(
        self,
        stellar_vec: np.ndarray,
        tce_features: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Use SHAP KernelExplainer on the combined feature vector.
        Wraps only the scalar feature inputs (stellar + TCE), not the LC views.
        """
        import shap
        import torch

        # Combine stellar + TCE into a flat input
        combined = np.concatenate([stellar_vec, tce_features]).reshape(1, -1).astype(np.float32)
        feature_names = STELLAR_FEATURE_NAMES + TRANSIT_FEATURE_NAMES

        def predict_fn(x: np.ndarray) -> np.ndarray:
            """Return TRANSIT class probability for a batch of feature vectors."""
            probs_out = np.zeros(x.shape[0], dtype=np.float32)
            with torch.no_grad():
                # Process sequentially to avoid 38GB OOM on large batches
                for i in range(x.shape[0]):
                    t = torch.from_numpy(x[i:i+1].astype(np.float32)).to(
                        next(self.model.parameters()).device
                    )
                    if hasattr(self.model, 'predict_from_features'):
                        probs = self.model.predict_from_features(t).cpu().numpy()
                    else:
                        probs = self._mock_predict(t).cpu().numpy()
                    probs_out[i] = probs[0, 0]
            return probs_out

        # Use 50 background samples (zeroed-out baseline)
        background = np.zeros((50, combined.shape[1]), dtype=np.float32)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(combined, nsamples=100, silent=True)

        results = []
        for i, name in enumerate(feature_names):
            raw_val = float(combined[0, i])
            sv = float(shap_vals[0, i]) if shap_vals is not None else 0.0
            results.append({"name": name, "value": raw_val, "shap_value": sv})

        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:10]

    def _mock_predict(self, t):
        """Safe fallback using only stellar branch of model."""
        import torch
        bs = t.shape[0]
        zeros_lc = torch.zeros(bs, 2001, device=t.device)
        zeros_local = torch.zeros(bs, 201, device=t.device)
        zeros_cv = torch.zeros(bs, 201, device=t.device)
        zeros_rf = torch.zeros(bs, 20000, device=t.device)
        out = self.model(zeros_rf, zeros_lc, zeros_local, t[:, :8], zeros_cv)
        return out["probs"]

    def _gradient_proxy(
        self,
        stellar_vec: np.ndarray,
        tce_features: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Fast gradient-based feature importance as SHAP fallback.
        Uses input × gradient as attribution proxy.
        """
        import torch

        combined = np.concatenate([stellar_vec, tce_features]).astype(np.float32)
        feature_names = STELLAR_FEATURE_NAMES + TRANSIT_FEATURE_NAMES

        t = torch.from_numpy(combined).unsqueeze(0).requires_grad_(True).to(
            next(self.model.parameters()).device
        )

        # Compute gradient of TRANSIT logit w.r.t. scalar features
        bs = 1
        zeros_lc = torch.zeros(bs, 2001, device=t.device)
        zeros_local = torch.zeros(bs, 201, device=t.device)
        zeros_cv = torch.zeros(bs, 201, device=t.device)
        zeros_rf = torch.zeros(bs, 20000, device=t.device)

        out = self.model(zeros_rf, zeros_lc, zeros_local, t[:, :8], zeros_cv)
        transit_logit = out["probs"][0, 0]  # TRANSIT probability
        transit_logit.backward()

        grads = t.grad[0].detach().cpu().numpy()
        importance = combined * grads  # input × gradient

        results = []
        for i, name in enumerate(feature_names):
            results.append({
                "name": name,
                "value": float(combined[i]),
                "shap_value": float(importance[i]),
            })

        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:10]


def explain_prediction(
    model,
    stellar_vec: np.ndarray,
    tce_features: np.ndarray,
    device=None,
) -> str:
    """
    Convenience wrapper — returns JSON string of top-10 SHAP features.
    Safe to call from the inference pipeline. Never raises.
    """
    try:
        explainer = SHAPExplainer(model=model, device=device)
        features = explainer.explain(stellar_vec, tce_features)
        return json.dumps(features)
    except Exception as e:
        logger.warning(f"XAI explain_prediction failed: {e}")
        return json.dumps([])
