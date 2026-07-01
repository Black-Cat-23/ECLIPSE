"""
Attention Visualization for ECLIPSE-PRIME.

Extracts transformer attention weights from the ECLIPSEPrime model output
and renders a compact heatmap image encoded as a base64 PNG string.

The frontend displays this as an overlay on the light curve chart.
"""
from __future__ import annotations

import base64
import io
from typing import Optional, List

import numpy as np
from loguru import logger


def render_attention_heatmap(
    attention_weights: Optional[List[float]],
    width: int = 400,
    height: int = 60,
    colormap: str = "plasma",
) -> Optional[str]:
    """
    Convert a 1D attention weight array into a base64-encoded PNG heatmap.

    Args:
        attention_weights: List of floats from model output["attention_weights"]
        width:  Output image width in pixels
        height: Output image height in pixels
        colormap: Matplotlib colormap name

    Returns:
        Base64-encoded PNG string (no data:image/png header), or None on failure.
    """
    if not attention_weights:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        weights = np.array(attention_weights, dtype=np.float32)
        # Normalize to [0, 1]
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights = np.ones_like(weights) * 0.5

        # Resize to target width by interpolation
        x_orig = np.linspace(0, 1, len(weights))
        x_new  = np.linspace(0, 1, width)
        weights_resized = np.interp(x_new, x_orig, weights)

        # Create heatmap: (height, width) array
        heatmap = np.tile(weights_resized, (height, 1))

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(
            heatmap,
            aspect="auto",
            cmap=colormap,
            interpolation="bilinear",
            vmin=0, vmax=1
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                    transparent=True, dpi=100)
        plt.close(fig)
        buf.seek(0)

        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return b64

    except Exception as e:
        logger.warning(f"Attention heatmap render failed: {e}")
        return None


def extract_attention_from_output(model_output: dict) -> Optional[List[float]]:
    """
    Pull attention weights from ECLIPSEPrime model output dict.
    Returns a flat list of floats or None if not present.
    """
    try:
        raw = model_output.get("attention_weights")
        if raw is None:
            return None
        if hasattr(raw, "tolist"):
            return raw.tolist()
        return list(raw)
    except Exception:
        return None

def render_centroid_map(width: int = 400, height: int = 300) -> Optional[str]:
    """Generates a mock astrometric centroid offset scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io, base64

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        # Generate background (out-of-transit) centroids
        x_out = np.random.normal(0, 1.2, 500)
        y_out = np.random.normal(0, 1.2, 500)
        
        # Generate in-transit centroids (slight offset or no offset for validation)
        x_in = np.random.normal(0.05, 1.0, 60)
        y_in = np.random.normal(-0.05, 1.0, 60)

        ax.scatter(x_out, y_out, c='#3B6A9A', alpha=0.3, s=15, label="Out of Transit")
        ax.scatter(x_in, y_in, c='#1FAD73', alpha=0.9, s=30, label="In Transit")
        
        ax.axhline(0, color='white', alpha=0.2, linestyle='--')
        ax.axvline(0, color='white', alpha=0.2, linestyle='--')

        # Formatting to look good on a dark UI
        ax.tick_params(colors='#BAE6FD', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#3B6A9A')
            spine.set_alpha(0.4)

        legend = ax.legend(loc="upper right", frameon=True, fontsize=8)
        legend.get_frame().set_facecolor('#0B172A')
        legend.get_frame().set_edgecolor('#3B6A9A')
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color('white')

        ax.set_xlabel("ΔRA (arcsec)", color="#BAE6FD", fontsize=9, labelpad=8)
        ax.set_ylabel("ΔDec (arcsec)", color="#BAE6FD", fontsize=9, labelpad=8)
        ax.set_title("Astrometric Centroid Offset", color="white", fontsize=10, pad=12, fontweight="bold")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Centroid map render failed: {e}")
        return None
