"""
Main training loop for ECLIPSE-PRIME.

Features:
  - Automatic Mixed Precision (AMP) via torch.cuda.amp — fits T4 15GB VRAM
  - Gradient clipping (max_norm=1.0) for stability
  - OneCycleLR scheduler with 10% warmup
  - Early stopping (patience=10 epochs on val macro F1)
  - Per-epoch multi-task metric logging
  - Checkpoint saving (last + best)

Usage:
    python -m src.training.train --config configs/default.yaml
    # or
    from src.training.train import train_eclipse
    train_eclipse(train_loader, val_loader, config)
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
from tqdm import tqdm

from src.models.eclipse_prime import ECLIPSEPrime
from src.training.losses import ECLIPSEMultiTaskLoss
from src.training.metrics import compute_all_metrics
from src.utils.checkpoint import save_checkpoint, load_checkpoint, get_best_checkpoint
from src.utils.config import ECLIPSEConfig, DEFAULT_CONFIG


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eclipse(
    train_loader,
    val_loader,
    config: Optional[ECLIPSEConfig] = None,
    resume_from: Optional[str] = None
):
    """
    Full training loop for ECLIPSE-PRIME.

    Args:
        train_loader:  DataLoader from make_eclipse_dataloader(..., split='train')
        val_loader:    DataLoader from make_eclipse_dataloader(..., split='val')
        config:        ECLIPSEConfig (defaults to DEFAULT_CONFIG)
        resume_from:   Path to checkpoint to resume from
    """
    cfg = config or DEFAULT_CONFIG
    set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ECLIPSE-PRIME on: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ECLIPSEPrime.from_config(cfg).to(device)
    param_counts = model.parameter_count()
    logger.info(f"ECLIPSE-PRIME parameters: {param_counts['total']:,} total "
                f"(A={param_counts['stream_a']:,}, B={param_counts['stream_b']:,}, "
                f"fusion={param_counts['fusion']:,}, heads={param_counts['heads']:,})")

    # ── Class weights for Focal Loss ──────────────────────────────────────────
    all_labels = train_loader.dataset.labels
    class_weights_np = compute_class_weight(
        "balanced", classes=np.arange(4), y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights_np).to(device)
    logger.info(f"Class weights: {dict(zip(['TRANSIT','EB','BLEND','OTHER'], class_weights_np.round(3)))}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = ECLIPSEMultiTaskLoss(
        class_weights=class_weights,
        physics_weight=cfg.training.physics_loss_weight
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg.training.epochs,
        pct_start=0.1,      # 10% warmup
        anneal_strategy="cos"
    )
    scaler = GradScaler(enabled=cfg.training.amp)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = payload.get("epoch", 0) + 1
        best_val_f1 = payload.get("metrics", {}).get("val_f1_macro", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best F1={best_val_f1:.4f}")

    patience_counter = 0
    checkpoint_dir = cfg.api.checkpoint_dir

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.epochs):
        t0 = time.time()

        # Training pass
        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg,
            is_train=True
        )

        # Validation pass
        val_metrics = _run_epoch(
            model, val_loader, criterion, optimizer, scheduler, scaler, device, cfg,
            is_train=False
        )

        elapsed = time.time() - t0
        val_f1 = val_metrics.get("f1_macro", 0.0)

        logger.info(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_metrics['loss_total']:.4f} | "
            f"val_loss={val_metrics['loss_total']:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_auc={val_metrics.get('auc_macro', 0):.4f} | "
            f"time={elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
            logger.info(f"New best val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(
            model, optimizer, epoch,
            metrics={"val_f1_macro": val_f1, **val_metrics},
            config_dict=cfg.to_dict(),
            checkpoint_dir=checkpoint_dir,
            is_best=is_best
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= cfg.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience_counter})")
            break

    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
    return best_val_f1


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: ECLIPSEMultiTaskLoss,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: ECLIPSEConfig,
    is_train: bool
) -> dict:
    """Run one training or validation epoch. Returns aggregated metrics dict."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    all_true_params, all_pred_params = [], []
    all_true_snr, all_pred_snr = [], []
    all_transit_mask = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
            # Move to device
            raw_flux = batch["raw_flux"].to(device)
            global_view = batch["global_view"].to(device)
            local_view = batch["local_view"].to(device)
            stellar = batch["stellar"].to(device)
            centroid = batch["centroid"].to(device)

            targets = {
                "class":      batch["class"].to(device),
                "period":     batch["period"].to(device),
                "duration":   batch["duration"].to(device),
                "depth":      batch["depth"].to(device),
                "snr":        batch["snr"].to(device),
                "has_params": batch["has_params"].to(device),
                "local_view": local_view,
            }

            # Forward pass with AMP
            with autocast(enabled=cfg.training.amp):
                outputs = model(raw_flux, global_view, local_view, stellar, centroid)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total"]

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Accumulate for metrics
            total_loss += loss.item()
            probs_np = outputs["probs"].detach().cpu().numpy()
            preds_np = np.argmax(probs_np, axis=1)
            labels_np = targets["class"].cpu().numpy()

            all_labels.extend(labels_np.tolist())
            all_preds.extend(preds_np.tolist())
            all_probs.append(probs_np)

            # Parameter metrics
            transit_mask_np = targets["has_params"].cpu().numpy()
            all_transit_mask.extend(transit_mask_np.tolist())
            all_true_params.append(np.stack([
                targets["period"].cpu().numpy(),
                targets["duration"].cpu().numpy(),
                targets["depth"].cpu().numpy()
            ], axis=1))
            all_pred_params.append(np.stack([
                outputs["period_mean"].detach().cpu().numpy(),
                outputs["duration_mean"].detach().cpu().numpy(),
                outputs["depth_mean"].detach().cpu().numpy()
            ], axis=1))
            all_true_snr.extend(targets["snr"].cpu().numpy().tolist())
            all_pred_snr.extend(outputs["snr_pred"].detach().cpu().numpy().tolist())

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_true_params_arr = np.concatenate(all_true_params, axis=0)
    all_pred_params_arr = np.concatenate(all_pred_params, axis=0)
    all_transit_arr = np.array(all_transit_mask, dtype=bool)

    metrics = compute_all_metrics(
        y_true_cls=all_labels_arr,
        y_pred_cls=all_preds_arr,
        y_probs=all_probs_arr,
        y_true_params=all_true_params_arr,
        y_pred_params=all_pred_params_arr,
        y_true_snr=np.array(all_true_snr),
        y_pred_snr=np.array(all_pred_snr),
        transit_mask=all_transit_arr
    )
    metrics["loss_total"] = total_loss / max(len(loader), 1)
    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECLIPSE-PRIME")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--train-csv", type=str, default="data/processed/tce_catalog_train.csv")
    parser.add_argument("--val-csv", type=str, default="data/processed/tce_catalog_val.csv")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    from src.training.data_loader import make_eclipse_dataloader

    cfg = ECLIPSEConfig.from_yaml(args.config) if args.config else DEFAULT_CONFIG
    train_loader = make_eclipse_dataloader(args.train_csv, config=cfg, split="train", augment=True)
    val_loader = make_eclipse_dataloader(args.val_csv, config=cfg, split="val", augment=False)

    train_eclipse(train_loader, val_loader, config=cfg, resume_from=args.resume)
