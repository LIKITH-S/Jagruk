"""
models/loader.py — Model registry: loads artefacts from disk at startup,
exposes them to the rest of the application via dependency injection.

Phase 3 will swap the placeholder _infer() with a real scikit-learn / XGBoost
pipeline call. No other file needs to change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib

from app.config import settings
from app.monitoring.metrics import MODEL_LOADED, MODEL_LOAD_DURATION_SECONDS
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Holds references to loaded model artefacts.

    Attributes
    ----------
    model          : The primary scikit-learn / XGBoost estimator (or None).
    preprocessor   : Fitted sklearn Pipeline / ColumnTransformer (or None).
    feature_names  : Ordered list of feature column names.
    version        : Human-readable version tag read from the artefact directory.
    is_ready       : True once all required artefacts are loaded successfully.
    """

    def __init__(self) -> None:
        self.model:         Any          = None
        self.preprocessor:  Any          = None
        self.feature_names: list[str]    = []
        self.version:       str          = "v0-placeholder"
        self.is_ready:      bool         = False

    # ── Startup ───────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Async-friendly loader — uses joblib in a thread pool via run_in_executor
        in Phase 3. Currently loads placeholder state synchronously."""
        t0 = time.perf_counter()
        artifacts_dir = Path(settings.MODEL_ARTIFACTS_DIR)

        model_path       = artifacts_dir / settings.PRIMARY_MODEL_FILENAME
        preprocessor_path = artifacts_dir / settings.PREPROCESSOR_FILENAME
        features_path    = artifacts_dir / settings.FEATURE_NAMES_FILENAME

        loaded_any = False

        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info("model_loaded", path=str(model_path))
            loaded_any = True
        else:
            # Phase 2 placeholder: model file not yet present
            logger.warning(
                "model_file_not_found",
                path=str(model_path),
                hint="Using placeholder inference until Phase 3 training is complete.",
            )

        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("preprocessor_loaded", path=str(preprocessor_path))

        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as fh:
                self.feature_names = json.load(fh)

        # Read version from a VERSION file in the artefacts dir if present
        version_path = artifacts_dir / "VERSION"
        if version_path.exists():
            self.version = version_path.read_text(encoding="utf-8").strip()
        else:
            self.version = "v1"

        elapsed = time.perf_counter() - t0
        MODEL_LOAD_DURATION_SECONDS.set(elapsed)

        # Mark ready regardless of whether real artefacts are present —
        # the service can still respond using placeholder logic in Phase 2.
        self.is_ready = True
        MODEL_LOADED.set(1)
        logger.info("model_registry_ready", version=self.version, elapsed_s=round(elapsed, 4))

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def unload(self) -> None:
        self.model        = None
        self.preprocessor = None
        self.is_ready     = False
        MODEL_LOADED.set(0)
        logger.info("model_registry_unloaded")


# Module-level singleton — shared across the FastAPI app lifetime
registry = ModelRegistry()
