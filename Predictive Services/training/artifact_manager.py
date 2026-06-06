"""
Manages persistence of all training artifacts:
  - model.pkl
  - model_meta.json
  - confusion_matrix.png
  - training_report.md
"""
import os
import json
import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Saves and loads all training artifacts."""

    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, model, filename="model.pkl"):
        path = os.path.join(self.artifacts_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", path)
        return path

    def load_model(self, filename="model.pkl"):
        path = os.path.join(self.artifacts_dir, filename)
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def save_meta(self, meta: dict, filename="model_meta.json"):
        path = os.path.join(self.artifacts_dir, filename)
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Model meta saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def save_confusion_matrix(self, cm: np.ndarray, labels=None, filename="confusion_matrix.png"):
        labels = labels or ["Normal", "Hazard"]
        path = os.path.join(self.artifacts_dir, filename)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # Training report
    # ------------------------------------------------------------------

    def save_training_report(
        self,
        model_name: str,
        metrics: dict,
        classification_report_str: str,
        feature_importances: dict = None,
        filename="training_report.md",
    ):
        path = os.path.join(self.artifacts_dir, filename)
        lines = [
            f"# Training Report – {model_name}",
            f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "\n## Performance Metrics\n",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in metrics.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

        lines += [
            "\n## Classification Report\n",
            "```",
            classification_report_str,
            "```",
        ]

        if feature_importances:
            lines += ["\n## Top Feature Importances\n", "| Feature | Importance |", "|---------|------------|"]
            sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_fi[:15]:
                lines.append(f"| {feat} | {imp:.4f} |")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Training report saved to %s", path)
        return path
