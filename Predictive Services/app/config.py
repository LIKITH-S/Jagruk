"""
config.py — Centralised configuration via environment variables.
All settings are read from the environment (or .env file in dev).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "predictive-service"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"              # development | staging | production
    LOG_LEVEL: str = "INFO"

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 2
    API_PREFIX: str = "/api/v1"

    # ── Model artefacts ──────────────────────────────────────────────────────
    MODEL_ARTIFACTS_DIR: str = str(BASE_DIR / "artifacts")
    PRIMARY_MODEL_FILENAME: str = "severity_model.joblib"
    PREPROCESSOR_FILENAME: str = "preprocessor.joblib"
    FEATURE_NAMES_FILENAME: str = "feature_names.json"

    # ── Fusion Engine integration ─────────────────────────────────────────────
    FUSION_ENGINE_URL: str = "http://fusion-engine:8001"
    FUSION_ENGINE_API_KEY: str = ""
    FUSION_ENGINE_TIMEOUT_SECONDS: int = 10

    # ── Redis (prediction cache) ──────────────────────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    CACHE_TTL_SECONDS: int = 300

    # ── Prometheus metrics ────────────────────────────────────────────────────
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # ── Drift detection ───────────────────────────────────────────────────────
    DRIFT_WINDOW_SIZE: int = 1000        # rolling sample size
    DRIFT_THRESHOLD: float = 0.05       # KS-test p-value threshold

    # ── Data paths ────────────────────────────────────────────────────────────
    DATA_DIR: str = str(BASE_DIR / "data")
    LOGS_DIR: str = str(BASE_DIR / "logs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
