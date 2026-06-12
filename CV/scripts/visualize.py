"""
Training Visualization Utilities for Jagruk CV.

Generates publication-quality training curves (loss, F1, accuracy) and
confusion matrix plots. Used by train.py and evaluate.py.

Usage (imported):
    from scripts.visualize import save_training_plots, save_confusion_matrix
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def save_training_plots(history: dict, output_dir: str = "reports"):
    """
    Generates and saves training visualization plots.

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'f1', 'accuracy'.
                 Each key maps to a list of per-epoch values.
        output_dir: Directory to save plot images.
    """
    # Lazy import so matplotlib doesn't slow down imports elsewhere
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        logger.warning("No training history to plot.")
        return

    # --- Style setup ---
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        "train": "#FF6B6B",
        "val": "#4ECDC4",
        "f1": "#45B7D1",
        "acc": "#96CEB4"
    }

    # --- 1. Loss Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], color=colors["train"],
            linewidth=2, marker='o', markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"], color=colors["val"],
            linewidth=2, marker='s', markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved to {loss_path}")

    # --- 2. F1-Macro Curve ---
    if "f1" in history and history["f1"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history["f1"], color=colors["f1"],
                linewidth=2, marker='D', markersize=4, label="F1-Macro")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("F1-Macro Score", fontsize=12)
        ax.set_title("Validation F1-Macro Score", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        f1_path = os.path.join(output_dir, "f1_curve.png")
        fig.savefig(f1_path, dpi=150)
        plt.close(fig)
        logger.info(f"F1 curve saved to {f1_path}")

    # --- 3. Accuracy Curve ---
    if "accuracy" in history and history["accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history["accuracy"], color=colors["acc"],
                linewidth=2, marker='^', markersize=4, label="Accuracy")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Validation Accuracy", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        acc_path = os.path.join(output_dir, "accuracy_curve.png")
        fig.savefig(acc_path, dpi=150)
        plt.close(fig)
        logger.info(f"Accuracy curve saved to {acc_path}")


def save_confusion_matrix(y_true: list, y_pred: list, class_names: list,
                          output_dir: str = "reports",
                          filename: str = "confusion_matrix.png"):
    """
    Generates and saves a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels (integer indices).
        y_pred: Predicted labels (integer indices).
        class_names: List of class name strings.
        output_dir: Directory to save the plot.
        filename: Output filename.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual", fontsize=12, fontweight='bold')
    ax.set_title("Confusion Matrix — Test Split", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out_path}")
