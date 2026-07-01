"""
Transformer Attention Rollout for Stream A visualization.
Aggregates multi-layer attention weights into per-timestep importance maps.

Reference: Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class AttentionRollout:
    """
    Extract and roll up attention weights from ECLIPSE-PRIME Stream A.

    Attention rollout: multiply attention matrices across layers,
    taking the average over heads. This gives a per-patch importance score
    that accounts for attention flow through the transformer depth.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._hooks = []
        self._attention_maps = []

    def _register_hooks(self) -> None:
        """Register forward hooks on all TransformerEncoderLayer modules."""
        self._attention_maps = []
        self._hooks = []

        for name, module in self.model.stream_a.transformer.layers.named_children():
            def hook(mod, inp, out, _name=name):
                # TransformerEncoderLayer with batch_first returns (output, attn_weights)
                # if need_weights=True. We access the self-attn submodule directly.
                pass
            # Use the self-attention submodule
            h = module.self_attn.register_forward_hook(
                lambda mod, inp, out: self._attention_maps.append(out[1])
            )
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_rollout(self, raw_flux: np.ndarray) -> np.ndarray:
        """
        Compute attention rollout for a raw flux sequence.

        Args:
            raw_flux: (T,) raw PDCSAP flux, will be padded to T_max

        Returns:
            importance: (n_patches,) attention importance per patch
        """
        T_max = self.model.stream_a.patch_embed.pos_embed.num_embeddings - 2
        padded = np.zeros(T_max, dtype=np.float32)
        padded[:min(len(raw_flux), T_max)] = raw_flux[:T_max]

        x = torch.from_numpy(padded).unsqueeze(0).to(self.device)

        self._register_hooks()
        self.model.stream_a.eval()
        with torch.no_grad():
            _ = self.model.stream_a(x)
        self._remove_hooks()

        if not self._attention_maps:
            # Fallback: uniform importance
            patches = self.model.stream_a.patch_embed(x)
            return np.ones(patches.shape[1])

        # Rollout: product of attention matrices (averaged over heads)
        rollout = None
        for attn in self._attention_maps:
            if attn is None:
                continue
            # attn: (B, n_heads, T, T) or (B, T, T)
            attn_np = attn[0].detach().cpu().numpy()
            if attn_np.ndim == 3:  # (heads, T, T)
                attn_avg = attn_np.mean(0)
            else:
                attn_avg = attn_np

            # Add residual identity connection
            n = attn_avg.shape[0]
            attn_aug = attn_avg + np.eye(n)
            attn_aug = attn_aug / (attn_aug.sum(axis=-1, keepdims=True) + 1e-8)

            rollout = attn_aug if rollout is None else attn_aug @ rollout

        if rollout is None:
            patches = self.model.stream_a.patch_embed(x)
            return np.ones(patches.shape[1])

        # Return column sum (importance of each patch position)
        importance = rollout.sum(0)
        importance = importance / (importance.max() + 1e-8)
        return importance.astype(np.float32)

    def importance_to_time(
        self, importance: np.ndarray, T: int, patch_size: int = 64, stride: int = 32
    ) -> np.ndarray:
        """
        Map patch-level importance back to original time cadences.
        Each patch covers [i*stride : i*stride + patch_size] cadences.

        Returns: (T,) per-cadence importance array.
        """
        time_importance = np.zeros(T, dtype=np.float32)
        for i, imp in enumerate(importance):
            start = i * stride
            end = min(start + patch_size, T)
            time_importance[start:end] += imp
        # Normalize
        mx = time_importance.max()
        if mx > 0:
            time_importance /= mx
        return time_importance
