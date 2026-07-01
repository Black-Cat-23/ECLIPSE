"""
PyTorch Dataset and DataLoader for ECLIPSE-PRIME training.

ECLIPSEDataset: loads from preprocessed .npz tensor files + label CSV.
ECLIPSEDataLoader: wraps with stratified sampler for class balance.

Each sample contains:
    raw_flux:       (T_max,)  float32 — padded PDCSAP flux for Stream A
    global_view:    (2001,)   float32 — TLS phase-fold global view
    local_view:     (201,)    float32 — TLS phase-fold local view
    centroid:       (201,)    float32 — phase-folded centroid displacement
    stellar:        (8,)      float32 — normalized stellar parameters
    targets:        dict with class, period, duration, depth, snr, has_params
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.utils.config import CLASS_TO_IDX, ECLIPSEConfig, DEFAULT_CONFIG


class ECLIPSEDataset(Dataset):
    """
    Dataset loading from preprocessed .npz tensor files.

    Each row in the catalog CSV corresponds to one TCE, with a path to the
    .npz file containing (raw_flux, global_view, local_view, centroid, stellar).
    Labels come from the 'eclipse_label' column.
    """

    def __init__(
        self,
        catalog_csv: str,
        config: Optional[ECLIPSEConfig] = None,
        augment: bool = False,
        split: str = "train"  # "train", "val", "test"
    ):
        self.config = config or DEFAULT_CONFIG
        self.augment = augment
        self.split = split

        df = pd.read_csv(catalog_csv)
        # Filter rows with valid tensor paths
        df = df[df["tensor_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

        if "eclipse_label" not in df.columns:
            df["eclipse_label"] = "OTHER"

        # Encode labels
        df["label_idx"] = df["eclipse_label"].map(CLASS_TO_IDX).fillna(3).astype(int)
        self.df = df
        self.labels = df["label_idx"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row["tensor_path"]

        # ── Load tensors ──────────────────────────────────────────────────────
        data = np.load(path, allow_pickle=False)
        raw_flux = data["raw_flux"].astype(np.float32)
        global_view = data["global_view"].astype(np.float32)
        local_view = data["local_view"].astype(np.float32)
        centroid = data["centroid"].astype(np.float32)
        stellar = data["stellar"].astype(np.float32)

        # ── Pad/truncate raw flux ─────────────────────────────────────────────
        T_max = self.config.model.T_max
        raw_flux = _pad_or_truncate(raw_flux, T_max)

        # ── Augmentation (train only) ─────────────────────────────────────────
        if self.augment and self.split == "train":
            raw_flux, global_view, local_view = _augment(
                raw_flux, global_view, local_view
            )

        # ── Targets ───────────────────────────────────────────────────────────
        label_idx = int(row["label_idx"])
        period = float(row.get("period", 0.0) or 0.0)
        duration = float(row.get("duration_days", 0.0) or 0.0)
        depth = float(row.get("depth", 0.0) or 0.0)
        snr = float(row.get("snr_tls", 0.0) or 0.0)
        has_params = bool(label_idx == 0 and period > 0)

        return {
            "raw_flux":      torch.from_numpy(raw_flux),
            "global_view":   torch.from_numpy(global_view),
            "local_view":    torch.from_numpy(local_view),
            "centroid":      torch.from_numpy(centroid),
            "stellar":       torch.from_numpy(stellar),
            "class":         torch.tensor(label_idx, dtype=torch.long),
            "period":        torch.tensor(period, dtype=torch.float32),
            "duration":      torch.tensor(duration, dtype=torch.float32),
            "depth":         torch.tensor(depth, dtype=torch.float32),
            "snr":           torch.tensor(snr, dtype=torch.float32),
            "has_params":    torch.tensor(has_params, dtype=torch.bool),
        }


def _pad_or_truncate(arr: np.ndarray, T_max: int) -> np.ndarray:
    if len(arr) >= T_max:
        return arr[:T_max]
    padded = np.zeros(T_max, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


def _augment(
    raw_flux: np.ndarray,
    global_view: np.ndarray,
    local_view: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    On-the-fly batch augmentations (applied per sample during training):
      1. Gaussian noise injection on raw flux (σ = 0.001)
      2. Random phase shift on global/local views (circular roll ±5%)
      3. Flux scaling ±5% (simulates dilution / contamination variation)
    """
    # Gaussian noise
    noise_scale = 0.001
    raw_flux = raw_flux + np.random.randn(*raw_flux.shape).astype(np.float32) * noise_scale

    # Random phase shift (circular roll)
    max_shift = int(0.05 * len(global_view))
    if max_shift > 0:
        shift = random.randint(-max_shift, max_shift)
        global_view = np.roll(global_view, shift)

    max_shift_local = int(0.05 * len(local_view))
    if max_shift_local > 0:
        shift_local = random.randint(-max_shift_local, max_shift_local)
        local_view = np.roll(local_view, shift_local)

    # Flux scaling
    scale = 1.0 + random.uniform(-0.05, 0.05)
    raw_flux = raw_flux * scale
    global_view = global_view * scale
    local_view = local_view * scale

    return raw_flux, global_view, local_view


def make_eclipse_dataloader(
    catalog_csv: str,
    config: Optional[ECLIPSEConfig] = None,
    split: str = "train",
    batch_size: Optional[int] = None,
    augment: bool = True,
    stratified: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Build a DataLoader for ECLIPSE training/validation.

    Uses WeightedRandomSampler for stratified oversampling of rare classes.
    """
    cfg = config or DEFAULT_CONFIG
    bs = batch_size or cfg.training.batch_size
    dataset = ECLIPSEDataset(catalog_csv, config=cfg, augment=(augment and split == "train"), split=split)

    if stratified and split == "train" and len(dataset) > 0:
        # Compute per-sample weights inversely proportional to class frequency
        class_counts = np.bincount(dataset.labels, minlength=4).astype(float)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[dataset.labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=bs,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        collate_fn=_collate_fn
    )


def _collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate: stack all keys. Handles dict-of-tensors format.
    'local_view' is also stacked separately for the physics loss.
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [b[k] for b in batch]
        collated[k] = torch.stack(vals)
    # Expose local_view in targets for physics loss
    collated["local_view_target"] = collated["local_view"].clone()
    return collated
