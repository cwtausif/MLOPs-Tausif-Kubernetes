# Custom Airflow image = official Airflow + our ML libraries
# Python 3.12 explicitly - matches the Flask image (same xgboost version)
FROM apache/airflow:3.3.1-python3.12

# XGBoost needs the OpenMP runtime (libgomp1)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Back to the normal airflow user for pip installs
USER airflow
# --chown/--chmod: make the file readable by the airflow user
COPY --chown=airflow:0 --chmod=644 requirements.txt /tmp/requirements.txt
# Pin Airflow's own version so pip doesn't upgrade/downgrade it by accident
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
