"""
Evaluation pipeline: ties metrics, confusion matrix and report generation together.
"""
import logging
from .metrics import compute_metrics, get_classification_report, get_confusion_matrix
from .artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a trained model and persists all evaluation artifacts."""

    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager

    def evaluate(self, trainer, X_test, y_test, feature_names=None):
        """
        Runs evaluation and persists confusion matrix + training report.

        Args:
            trainer: A fitted trainer (RandomForestTrainer or XGBoostTrainer).
            X_test:  Test feature matrix.
            y_test:  True test labels.
            feature_names (list, optional): Column names for feature importance.

        Returns:
            dict: Evaluation metrics.
        """
        model_name = trainer.MODEL_NAME

        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)

        # --- Metrics ---
        metrics = compute_metrics(y_test, y_pred, y_prob)
        logger.info("[%s] Metrics: %s", model_name, metrics)

        # --- Classification report ---
        clf_report = get_classification_report(y_test, y_pred)
        logger.info("[%s] Classification Report:\n%s", model_name, clf_report)

        # --- Confusion matrix ---
        cm = get_confusion_matrix(y_test, y_pred)
        self.artifact_manager.save_confusion_matrix(cm)

        # --- Feature importances ---
        feature_importances = None
        if feature_names is not None:
            try:
                feature_importances = trainer.get_feature_importances(feature_names)
            except Exception as e:
                logger.warning("Could not extract feature importances: %s", e)

        # --- Training report ---
        self.artifact_manager.save_training_report(
            model_name=model_name,
            metrics=metrics,
            classification_report_str=clf_report,
            feature_importances=feature_importances,
        )

        return metrics
