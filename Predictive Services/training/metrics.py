"""
Metric computation utilities for the training pipeline.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes a full set of classification metrics.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for positive class (for AUC).

    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def get_classification_report(y_true, y_pred, target_names=None):
    """Returns sklearn classification report as a string."""
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names or ["Normal", "Hazard"],
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred):
    """Returns confusion matrix as a numpy array."""
    return confusion_matrix(y_true, y_pred)
