"""
ExoMiner Adapter — optional pretrained weight initialization for Stream B.

If ExoMiner pretrained weights are available, this adapter transfers
compatible CNN encoder weights to ECLIPSE-PRIME's LightCurveCNN in Stream B.
Gracefully skips if weights are not found.

ExoMiner reference: Valizadegan et al. (2022) ApJ 926, 120.
Note: ExoMiner++ is binary (PC vs FP). We use it only for CNN encoder
pretraining, not for classification weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


def load_exominer_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False
) -> bool:
    """
    Attempt to load ExoMiner pretrained weights into an ECLIPSE-PRIME model.

    Uses non-strict loading: only weights whose names and shapes match are
    transferred. Mismatched layers (e.g., classification head) are skipped.

    Args:
        model:        ECLIPSEPrime model instance
        weights_path: Path to ExoMiner .pt or .pth checkpoint
        strict:       If False, allow partial weight loading

    Returns:
        True if any weights were successfully loaded, False otherwise.
    """
    path = Path(weights_path)
    if not path.exists():
        logger.info(f"ExoMiner weights not found at {weights_path}. "
                    "Training from scratch.")
        return False

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        # ExoMiner checkpoints may store weights under different keys
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # ── Attempt to map ExoMiner conv layers to Stream B CNN ──────────────
        # ExoMiner typically uses: global_conv_1 ... global_conv_n
        # We map these to stream_b.global_cnn.blocks.0 ... etc.
        mapped_keys = 0
        model_state = model.state_dict()
        transferred = {}

        for ext_key, ext_val in state_dict.items():
            # Try direct match first
            if ext_key in model_state:
                if model_state[ext_key].shape == ext_val.shape:
                    transferred[ext_key] = ext_val
                    mapped_keys += 1
            else:
                # Try mapping ExoMiner key patterns to Stream B
                mapped_key = _map_exominer_key(ext_key)
                if mapped_key and mapped_key in model_state:
                    if model_state[mapped_key].shape == ext_val.shape:
                        transferred[mapped_key] = ext_val
                        mapped_keys += 1

        if mapped_keys > 0:
            model.load_state_dict({**model_state, **transferred}, strict=False)
            logger.info(f"ExoMiner adapter: transferred {mapped_keys} weight tensors "
                        f"from {weights_path}")
            return True
        else:
            logger.warning("ExoMiner adapter: no compatible weights found. "
                           "Training from scratch.")
            return False

    except Exception as e:
        logger.warning(f"ExoMiner adapter load failed: {e}. Training from scratch.")
        return False


def _map_exominer_key(ext_key: str) -> Optional[str]:
    """
    Map ExoMiner checkpoint key names to ECLIPSE-PRIME Stream B keys.
    Returns None if no mapping is found.
    """
    # Common ExoMiner key patterns → ECLIPSE-PRIME paths
    mappings = {
        "global_conv_layers.0": "stream_b.global_cnn.blocks.0.conv",
        "global_conv_layers.1": "stream_b.global_cnn.blocks.1.conv",
        "local_conv_layers.0": "stream_b.local_mha_cnn.cnn.0.conv",
        "local_conv_layers.1": "stream_b.local_mha_cnn.cnn.1.conv",
    }
    for ext_pattern, eclipse_prefix in mappings.items():
        if ext_key.startswith(ext_pattern):
            suffix = ext_key[len(ext_pattern):]
            return f"{eclipse_prefix}{suffix}"
    return None


def freeze_stream_b_cnn(model: nn.Module, n_frozen_blocks: int = 2) -> None:
    """
    Freeze the first n_frozen_blocks of Stream B's global CNN.
    Useful when fine-tuning pretrained ExoMiner weights on TESS data.
    """
    for i, block in enumerate(model.stream_b.global_cnn.blocks):
        if i < n_frozen_blocks:
            for param in block.parameters():
                param.requires_grad = False
            logger.info(f"Frozen Stream B global CNN block {i}")
