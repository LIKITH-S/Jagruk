# Predictive Service

> **Disaster Alert Ecosystem — Predictive Modeling Service**
> Predicts future disaster severity from environmental and satellite signals and delivers structured forecasts to the Fusion Engine.

---

## Overview

| Attribute | Value |
|-----------|-------|
| Language | Python 3.11 |
| Framework | FastAPI |
| Core ML | XGBoost · scikit-learn · LightGBM |
| Explainability | SHAP |
| Cache | Redis |
| Metrics | Prometheus |
| Containerisation | Docker (multi-stage) |

---

## Folder Structure

```
predictive-service/
│
├── app/                        # Application source
│   ├── main.py                 # FastAPI app factory & health endpoint
│   ├── config.py               # Pydantic Settings (env-var driven)
│   ├── api/                    # Route handlers (Phase 2)
│   ├── services/               # Business logic layer (Phase 2)
│   ├── models/                 # Model loader & registry (Phase 2)
│   ├── preprocessing/          # Input cleaning & normalisation pipelines
│   ├── feature_engineering/    # Derived feature construction
│   ├── explainability/         # SHAP value generation
│   ├── monitoring/             # Prometheus metrics & drift detection
│   ├── utils/                  # Logging, HTTP clients, helpers
│   └── schemas/                # Pydantic request/response schemas
│
├── training/                   # Offline training scripts (Phase 3)
├── tests/                      # pytest test suite
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── scripts/                    # Ops scripts (model promotion, data export …)
├── artifacts/                  # Trained model files (.joblib) — git-ignored
├── data/                       # Raw & processed datasets — git-ignored
│   ├── raw/
│   └── processed/
├── logs/                       # Structured JSON logs — git-ignored
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Docker ≥ 24 and Docker Compose ≥ 2.24
- Python 3.11 (for local dev without Docker)

### 2. Environment

```bash
cp .env.example .env
# Edit .env — set FUSION_ENGINE_API_KEY and REDIS_PASSWORD at minimum
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

| Endpoint | URL |
|----------|-----|
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Prometheus metrics | http://localhost:9090/metrics |

### 4. Run locally (dev)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## Configuration Reference

All settings are loaded from environment variables (or `.env` in dev).
See [`.env.example`](.env.example) for the full list with descriptions.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENV` | `development` | Runtime environment |
| `LOG_LEVEL` | `INFO` | Structlog level |
| `API_PORT` | `8000` | FastAPI listen port |
| `MODEL_ARTIFACTS_DIR` | `/app/artifacts` | Path to `.joblib` files |
| `FUSION_ENGINE_URL` | `http://fusion-engine:8001` | Upstream integration URL |
| `REDIS_HOST` | `redis` | Cache host |
| `DRIFT_THRESHOLD` | `0.05` | KS-test p-value for drift alerts |

---

## Development Phases

| Phase | Status | Scope |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Scaffold, config, Docker, env |
| Phase 2 | 🔜 Next | API routes, prediction service, schemas, caching |
| Phase 3 | ⏳ Planned | Training pipeline, feature engineering, SHAP |
| Phase 4 | ⏳ Planned | Drift monitoring, Prometheus metrics, alerting |

---

## Testing

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Deployment Notes

- The Docker image runs as a **non-root user** (`appuser`, UID 1001).
- Model artefacts are mounted via a named Docker volume (`predictive-artifacts`) so they survive container restarts.
- Attach to the `disaster-net` bridge network to communicate with the Fusion Engine and other ecosystem services.
- Set `API_WORKERS` to `(2 × CPU cores) + 1` for production throughput.

---

## Licence

Internal — Disaster Alert Platform. Not for public distribution.
