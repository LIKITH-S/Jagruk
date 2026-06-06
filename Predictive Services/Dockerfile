# =============================================================================
# Dockerfile — Predictive Service
# Multi-stage build: builder installs deps, runtime image is lean.
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed only at build time (compilers for native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="platform-team@disaster-alert.io"
LABEL service="predictive-service"

# libgomp1 required by XGBoost / LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/         ./app/
COPY artifacts/   ./artifacts/
COPY data/        ./data/
COPY scripts/     ./scripts/

# Logs and data dirs — writable by appuser
RUN mkdir -p logs \
    && chown -R appuser:appgroup /app

USER appuser

# Expose API and Prometheus metrics ports
EXPOSE 8000 9090

# Health-check so Docker / orchestrators can verify liveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server: uvicorn with multiple workers
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]
