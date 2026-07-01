"""Evaluation suite — benchmark, confusion matrix, calibration, injection test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


CLASS_NAMES = ["TRANSIT", "EB", "BLEND", "OTHER"]


def plot_4class_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "data/reports/confusion_matrix.png",
    normalize: bool = True
) -> str:
    """Plot and save a 4-class confusion matrix."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("ECLIPSE-PRIME 4-Class Confusion Matrix", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {output_path}")
    return output_path


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: str = "data/reports/reliability_diagram.png",
    n_bins: int = 10
) -> str:
    """Reliability diagram for all 4 classes."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for cls_idx, (cls_name, ax) in enumerate(zip(CLASS_NAMES, axes)):
        probs_cls = y_probs[:, cls_idx]
        labels_cls = (y_true == cls_idx).astype(int)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_accs = [], []
        for i in range(n_bins):
            mask = (probs_cls >= bins[i]) & (probs_cls < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_accs.append(labels_cls[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(bin_centers, bin_accs, "o-", color="#1565c0", label=f"{cls_name}")
        ax.fill_between(bin_centers, bin_accs, bin_centers[:len(bin_accs)],
                        alpha=0.2, color="orange", label="Miscalibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{cls_name} Reliability Diagram")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("ECLIPSE-PRIME Calibration (Reliability Diagrams)", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Reliability diagram saved: {output_path}")
    return output_path


def run_injection_recovery_test(
    pipeline,
    clean_time: np.ndarray,
    clean_flux: np.ndarray,
    n_injections: int = 200,
    snr_bins: int = 10
) -> pd.DataFrame:
    """
    Injection-recovery test: inject known transits, run pipeline, check recovery.

    Returns DataFrame with columns: true_period, true_depth, true_snr,
    recovered, recovered_period, period_error_frac, class_pred.
    """
    from src.preprocessing.injection_recovery import inject_transit, generate_injection_batch

    records = []
    batch = generate_injection_batch(clean_time, clean_flux, n_injections=n_injections)

    for sample in batch:
        true_period = sample["true_period"]
        true_depth = sample["true_depth"]
        flux_inj = sample["flux_injected"]

        # Mock inference using TLS on injected flux
        try:
            from src.preprocessing.period_search import run_tls_search
            tces = run_tls_search(clean_time, flux_inj, sde_threshold=5.0)

            if tces:
                best = max(tces, key=lambda t: t.snr)
                period_err = abs(best.period - true_period) / true_period
                recovered = period_err < 0.05  # within 5%
            else:
                best = None
                period_err = None
                recovered = False

            records.append({
                "true_period": true_period,
                "true_depth": true_depth,
                "true_duration": sample["true_duration"],
                "snr_tls": best.snr if best else 0.0,
                "recovered": recovered,
                "recovered_period": best.period if best else None,
                "period_error_frac": period_err
            })
        except Exception as e:
            logger.warning(f"Injection recovery step failed: {e}")

    return pd.DataFrame(records)
