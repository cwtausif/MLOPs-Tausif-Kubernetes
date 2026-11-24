"""
config.py — all paths and settings in one place.

Paths come from environment variables so the same code works:
  - on your Mac         (defaults → ./data and ./models)
  - in Airflow          (DATA_DIR=/opt/airflow/data, MODEL_DIR=/opt/airflow/models)
  - in Kubernetes       (MODEL_DIR=/app/models)
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))

# --- data files ---
RAW_JSON = DATA_DIR / "raw.json"  # output of etl_fetch.py
CLEAN_JSON = DATA_DIR / "clean.json"  # output of preprocess.py

# --- candidate (new model, not live yet) ---
CANDIDATE_MODEL = MODEL_DIR / "candidate.pkl"
CANDIDATE_METRICS = MODEL_DIR / "candidate_metrics.json"
CANDIDATE_PREDICTIONS = MODEL_DIR / "candidate_predictions.json"

# --- live model (used by Flask) ---
LIVE_MODEL = MODEL_DIR / "model.pkl"
LIVE_METRICS = MODEL_DIR / "metrics.json"
LIVE_PREDICTIONS = MODEL_DIR / "predictions.json"

# --- settings ---
TICKER = os.getenv("TICKER", "BTC-USD")
START_DATE = os.getenv("START_DATE", "2017-01-01")
FORECAST_DAYS = 7  # predict next 7 days
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "10"))  # % — max error allowed

# Flask endpoint that reloads the model (used by notify.py)
FLASK_RELOAD_URL = os.getenv("FLASK_RELOAD_URL", "http://localhost:5000/reload")


def ensure_dirs():
    """Create data/ and models/ folders if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
