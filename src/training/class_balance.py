"""
Class balancing utilities for ECLIPSE training.
Provides Focal Loss integration and SMOTE-based oversampling for rare BLEND class.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger


def compute_eclipse_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute balanced class weights for Focal Loss.
    Uses sklearn's balanced strategy: weight = N / (n_classes × count_per_class)

    Returns: np.float32 (4,) — weights for [TRANSIT, EB, BLEND, OTHER]
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=np.arange(4), y=labels)
    logger.info(f"Class weights: TRANSIT={weights[0]:.3f}, EB={weights[1]:.3f}, "
                f"BLEND={weights[2]:.3f}, OTHER={weights[3]:.3f}")
    return weights.astype(np.float32)


def smote_oversample_features(
    features: np.ndarray,
    labels: np.ndarray,
    target_class: int = 2,    # BLEND = class 2
    strategy: str = "minority"
) -> tuple:
    """
    Apply SMOTE oversampling to rare classes in the feature space.
    Operates on flattened feature vectors (not on raw flux directly).

    Args:
        features:     (N, D) feature matrix
        labels:       (N,) integer class labels
        target_class: Class index to oversample (default 2 = BLEND)
        strategy:     SMOTE sampling strategy

    Returns:
        (features_resampled, labels_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=3)
        features_res, labels_res = sm.fit_resample(features, labels)
        logger.info(f"SMOTE: {len(labels)} → {len(labels_res)} samples")
        return features_res, labels_res
    except ImportError:
        logger.warning("imbalanced-learn not available. Skipping SMOTE.")
        return features, labels
    except Exception as e:
        logger.warning(f"SMOTE failed: {e}. Returning original data.")
        return features, labels
