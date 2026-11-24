# =============================================================================
# app/Dockerfile - production image for the Flask prediction API (Part B1)
#
# Multi-stage build:
#   stage 1 "builder" -> installs Python libraries (has pip caches, build tools)
#   stage 2 "runtime" -> copies ONLY the installed libraries + our code
#                       -> smaller, cleaner final image
#
# Build from the project root (so src/ can be copied too):
#   docker build -f app/Dockerfile -t btc-forecast-api .
# =============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build
# Copy requirements first -> this layer is cached until requirements change
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# libgomp1 = OpenMP runtime that XGBoost needs; curl is NOT installed (smaller image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

# Installed libraries from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app
# Our code: the API + the shared feature code from src/
COPY --chown=app:app src/__init__.py src/config.py src/preprocess.py ./src/
COPY --chown=app:app app/app.py ./

# Model folder (mounted from a volume / Kubernetes PVC at runtime)
RUN mkdir -p /app/models && chown app:app /app/models

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MODEL_DIR=/app/models \
    PORT=5000

USER app
EXPOSE 5000

# Docker-level health check (Kubernetes uses its own probes instead)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:' + os.environ['PORT'] + '/health', timeout=4)"

# Gunicorn = production WSGI server (Part A3)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS:-2} --timeout 60 --access-logfile - app:app"]
