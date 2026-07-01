"""
Metrics for ECLIPSE-PRIME evaluation.

Classification metrics:
  - Per-class F1, precision, recall
  - Macro F1 (primary metric for model selection)
  - AUC-ROC (one-vs-rest for each class)
  - Balanced accuracy

Parameter regression metrics:
  - RMSE for period, duration, depth (TRANSIT-only)
  - MAE for period, duration, depth

SNR metrics:
  - MAE between predicted and TLS SNR

Conformal coverage:
  - Empirical coverage vs target coverage
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score,
    confusion_matrix
)

CLASS_NAMES = ["TRANSIT", "EB", "BLEND", "OTHER"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, float]:
    """
    Compute full classification metric suite.

    Args:
        y_true:   (N,) true class indices
        y_pred:   (N,) predicted class indices
        y_probs:  (N, 4) softmax probabilities

    Returns:
        dict with f1_macro, f1_transit, f1_eb, f1_blend, f1_other,
        auc_macro, balanced_acc
    """
    metrics = {}

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"f1_{name.lower()}"] = float(f1_per[i])

    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class precision / recall
    prec_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        metrics[f"prec_{name.lower()}"] = float(prec_per[i])
        metrics[f"rec_{name.lower()}"] = float(rec_per[i])

    # AUC-ROC (one-vs-rest)
    try:
        if len(np.unique(y_true)) > 1:
            metrics["auc_macro"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
        else:
            metrics["auc_macro"] = 0.0
    except Exception:
        metrics["auc_macro"] = 0.0

    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))

    return metrics


def compute_parameter_metrics(
    y_true_params: np.ndarray,   # (N, 3) [period, duration, depth]
    y_pred_params: np.ndarray,   # (N, 3)
    transit_mask: Optional[np.ndarray] = None  # (N,) bool
) -> Dict[str, float]:
    """
    Compute RMSE and MAE for transit parameter regression.
    Only computed on TRANSIT-class samples.

    Returns:
        dict with period_rmse, period_mae, duration_rmse, duration_mae,
        depth_rmse, depth_mae
    """
    metrics = {}
    param_names = ["period", "duration", "depth"]

    if transit_mask is not None and transit_mask.sum() == 0:
        for name in param_names:
            metrics[f"{name}_rmse"] = float("nan")
            metrics[f"{name}_mae"] = float("nan")
        return metrics

    if transit_mask is not None:
        true = y_true_params[transit_mask]
        pred = y_pred_params[transit_mask]
    else:
        true = y_true_params
        pred = y_pred_params

    for i, name in enumerate(param_names):
        diff = true[:, i] - pred[:, i]
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        mae = float(np.nanmean(np.abs(diff)))
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_mae"] = mae

    return metrics


def compute_snr_metrics(
    y_true_snr: np.ndarray,
    y_pred_snr: np.ndarray,
    transit_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute SNR regression metrics."""
    if transit_mask is not None:
        if transit_mask.sum() == 0:
            return {"snr_mae": float("nan"), "snr_rmse": float("nan")}
        true = y_true_snr[transit_mask]
        pred = y_pred_snr[transit_mask]
    else:
        true, pred = y_true_snr, y_pred_snr

    diff = true - pred
    return {
        "snr_mae": float(np.nanmean(np.abs(diff))),
        "snr_rmse": float(np.sqrt(np.nanmean(diff ** 2)))
    }


def compute_all_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_probs: np.ndarray,
    y_true_params: Optional[np.ndarray] = None,
    y_pred_params: Optional[np.ndarray] = None,
    y_true_snr: Optional[np.ndarray] = None,
    y_pred_snr: Optional[np.ndarray] = None,
    transit_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all ECLIPSE evaluation metrics in one call.
    Returns a flat dict suitable for logging to loguru / W&B.
    """
    all_metrics = {}
    all_metrics.update(compute_classification_metrics(y_true_cls, y_pred_cls, y_probs))

    if y_true_params is not None and y_pred_params is not None:
        all_metrics.update(compute_parameter_metrics(y_true_params, y_pred_params, transit_mask))

    if y_true_snr is not None and y_pred_snr is not None:
        all_metrics.update(compute_snr_metrics(y_true_snr, y_pred_snr, transit_mask))

    return all_metrics
