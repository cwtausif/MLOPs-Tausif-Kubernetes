# Image used by every Kubeflow container component (fetch / features / train / evaluate).
# Build from the project root:
#   docker build -f kubeflow/Dockerfile -t btc-pipeline .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home app

WORKDIR /app
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --chown=app:app src/ ./src/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER app
