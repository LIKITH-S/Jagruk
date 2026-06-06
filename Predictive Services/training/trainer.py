"""
Trainer classes for RandomForest and XGBoost models.
Each trainer exposes a consistent .fit() / .predict() interface.
"""
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Wraps RandomForestClassifier with a clean train/predict API."""

    MODEL_NAME = "RandomForest"

    def __init__(self, n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X_train, y_train):
        logger.info("Training %s on %d samples …", self.MODEL_NAME, len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("%s training complete.", self.MODEL_NAME)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))

    def get_params(self):
        return self.model.get_params()


class XGBoostTrainer:
    """Wraps XGBClassifier with a clean train/predict API."""

    MODEL_NAME = "XGBoost"

    def __init__(
        self,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=None,
        random_state=42,
        **kwargs,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            **kwargs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        logger.info("Training %s on %d samples …", self.MODEL_NAME, len(X_train))
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("%s training complete.", self.MODEL_NAME)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names):
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))

    def get_params(self):
        return self.model.get_params()
