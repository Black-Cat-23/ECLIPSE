"""
Model checkpoint save/load utilities.
Saves: model state, optimizer state, epoch, best metric, config snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config_dict: dict,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None,
    is_best: bool = False
) -> str:
    """
    Save a training checkpoint.
    
    Returns the path to the saved checkpoint.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    fname = filename or f"eclipse_epoch{epoch:04d}.pt"
    path = Path(checkpoint_dir) / fname

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config_dict
    }
    torch.save(payload, path)
    logger.info(f"Checkpoint saved: {path} | metrics={metrics}")

    if is_best:
        best_path = Path(checkpoint_dir) / "best.pt"
        torch.save(payload, best_path)
        logger.info(f"New best checkpoint: {best_path}")

    return str(path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Load a checkpoint into model (and optionally optimizer).
    
    Returns the checkpoint dict (contains epoch, metrics, config).
    """
    device = device or torch.device("cpu")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    logger.info(f"Model loaded from {checkpoint_path} (epoch {payload['epoch']})")

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    return payload


def get_best_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """Return path to best.pt if it exists, else None."""
    p = Path(checkpoint_dir) / "best.pt"
    return str(p) if p.exists() else None
