"""
Captum Integrated Gradients for Stream B CNN attribution.
Attributes classification predictions to phase bins in global/local views.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class IntegratedGradientsExplainer:
    """
    Captum IG on Stream B CNN.
    Shows which phase-folded light curve features drive classification.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._ig = None
        self._init_ig()

    def _init_ig(self) -> None:
        try:
            from captum.attr import IntegratedGradients

            class StreamBWrapper(nn.Module):
                """Wraps Stream B + heads for IG attribution on local_view."""
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, local_view, global_view, stellar, centroid):
                    feat, _ = self.m.stream_b(global_view, local_view, stellar, centroid)
                    feat_a = self.m.stream_a(
                        torch.zeros(local_view.shape[0], self.m.stream_a.patch_embed.pos_embed.num_embeddings - 2,
                                    device=local_view.device)
                    )
                    fused = self.m.fusion(feat_a, feat)
                    return self.m.head.clf(fused)["logits"]

            self._wrapper = StreamBWrapper(self.model).to(self.device)
            self._wrapper.eval()
            self._ig = IntegratedGradients(self._wrapper)
            logger.info("Captum IG explainer initialized")

        except ImportError:
            logger.warning("captum not installed. IG explainer disabled.")
        except Exception as e:
            logger.warning(f"IG init failed: {e}")

    def attribute(
        self,
        local_view: np.ndarray,
        global_view: np.ndarray,
        stellar: np.ndarray,
        centroid: np.ndarray,
        target_class: int = 0  # 0=TRANSIT
    ) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients attribution.

        Returns:
            dict with local_view_ig (201,), global_view_ig (2001,)
        """
        if self._ig is None:
            return {"local_view_ig": np.zeros(201), "global_view_ig": np.zeros(2001)}

        try:
            lv = torch.from_numpy(local_view).unsqueeze(0).float().to(self.device).requires_grad_(True)
            gv = torch.from_numpy(global_view).unsqueeze(0).float().to(self.device)
            sp = torch.from_numpy(stellar).unsqueeze(0).float().to(self.device)
            cv = torch.from_numpy(centroid).unsqueeze(0).float().to(self.device)

            baseline_lv = torch.zeros_like(lv)

            attrs = self._ig.attribute(
                lv,
                baselines=baseline_lv,
                target=target_class,
                additional_forward_args=(gv, sp, cv),
                n_steps=50
            )
            return {
                "local_view_ig": attrs[0].detach().cpu().numpy(),
                "global_view_ig": np.zeros(2001)  # IG on local only
            }
        except Exception as e:
            logger.warning(f"IG attribute failed: {e}")
            return {"local_view_ig": np.zeros(201), "global_view_ig": np.zeros(2001)}
