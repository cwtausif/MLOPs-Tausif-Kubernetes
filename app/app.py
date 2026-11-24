"""
app.py — Flask REST API that serves the 7-day BTC price forecast.

Endpoints
---------
GET  /health    → is the app alive? uptime + loaded model version
POST /predict   → next N days (1–7) of predicted closing prices
                  body (optional): {"forecast_days": 7}
                  or custom history:
                  {"history": [{"date", "open", "high", "low", "close", "volume"}, ...]}
GET  /metrics   → MAE / RMSE / MAPE from the latest validation run
POST /reload    → re-read model.pkl from disk (called by Airflow after retraining)

The model file lives in MODEL_DIR (/app/models in Docker/Kubernetes).
The app also checks the file's timestamp on each request, so EVERY worker
and EVERY pod picks up a new model automatically — /reload just forces it now.

Run (production):  gunicorn --bind 0.0.0.0:5000 app:app
"""

import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.preprocess import FEATURE_COLUMNS, add_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")
MAX_FORECAST_DAYS = 7
MIN_HISTORY_ROWS = 40  # features need 30 days of history + a few extra

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("btc-api")

app = Flask(__name__)
START_TIME = time.time()


# ---------------------------------------------------------------------------
# Model loading (thread-safe, auto-reloads when model.pkl changes)
# ---------------------------------------------------------------------------
class ModelStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.bundle = None
        self.mtime = None
        self.loaded_at = None
        self.error = None

    @property
    def path(self) -> Path:
        return MODEL_DIR / "model.pkl"

    def load(self) -> bool:
        """(Re)load model.pkl from disk. Returns True if a model is loaded."""
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
                with open(self.path, "rb") as f:
                    bundle = pickle.load(f)
                # one thread per prediction: pods have a small CPU limit, and
                # OpenMP would otherwise start one thread per *host* CPU
                for model in bundle.get("models", {}).values():
                    model.set_params(n_jobs=1)
                self.bundle, self.mtime = bundle, mtime
                self.loaded_at = datetime.now(timezone.utc).isoformat()
                self.error = None
                log.info("Loaded model %s from %s", bundle.get("version"), self.path)
                return True
            except FileNotFoundError:
                self.error = f"model file not found: {self.path}"
            except Exception as e:  # corrupt file, wrong format, ...
                self.error = f"failed to load model: {e}"
            log.warning(self.error)
            return self.bundle is not None

    def get(self):
        """Return the current model, reloading first if the file on disk changed."""
        try:
            if self.bundle is None or self.path.stat().st_mtime != self.mtime:
                self.load()
        except FileNotFoundError:
            pass  # keep serving the model already in memory, if any
        return self.bundle

    @property
    def version(self):
        return self.bundle.get("version") if self.bundle else None


store = ModelStore()
store.load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class BadRequest(Exception):
    """Raised for invalid user input → HTTP 400."""


def require_model():
    bundle = store.get()
    if bundle is None:
        return None, error_response(503, "Model not available", store.error)
    return bundle, None


def error_response(status: int, message: str, detail=None):
    body = {"error": message, "status": status}
    if detail:
        body["detail"] = detail
    return jsonify(body), status


def parse_forecast_days(payload: dict) -> int:
    value = payload.get("forecast_days", MAX_FORECAST_DAYS)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest("'forecast_days' must be an integer")
    if not 1 <= value <= MAX_FORECAST_DAYS:
        raise BadRequest(f"'forecast_days' must be between 1 and {MAX_FORECAST_DAYS}")
    return value


def history_to_features(history) -> tuple:
    """Validate custom history and return (last_date, last_close, feature_row)."""
    if not isinstance(history, list) or len(history) < MIN_HISTORY_ROWS:
        raise BadRequest(
            f"'history' must be a list of at least {MIN_HISTORY_ROWS} daily records"
        )
    required = ["date", "open", "high", "low", "close", "volume"]
    for i, row in enumerate(history):
        if not isinstance(row, dict):
            raise BadRequest(f"history[{i}] must be an object")
        missing = [k for k in required if row.get(k) is None]
        if missing:
            raise BadRequest(f"history[{i}] is missing: {', '.join(missing)}")
        for k in required[1:]:
            v = row[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise BadRequest(f"history[{i}].{k} must be a non-negative number")
        if row["close"] <= 0:
            raise BadRequest(f"history[{i}].close must be > 0")

    df = pd.DataFrame(history)[required]
    try:
        df["date"] = pd.to_datetime(df["date"])
    except (ValueError, TypeError):
        raise BadRequest("history dates must be valid dates, e.g. '2026-11-17'")
    df = (
        df.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    feats = add_features(df)
    if feats.empty:
        raise BadRequest("not enough history to compute features")
    last = feats.iloc[-1]
    return (
        last["date"],
        float(last["close"]),
        last[FEATURE_COLUMNS].to_numpy(dtype=float),
    )


def make_forecast(bundle, features, last_close, last_date, days):
    x = np.asarray(features, dtype=float).reshape(1, -1)
    out = []
    for h in range(1, days + 1):
        log_return = float(bundle["models"][h].predict(x)[0])
        out.append(
            {
                "date": (last_date + timedelta(days=h)).strftime("%Y-%m-%d"),
                "close": round(last_close * float(np.exp(log_return)), 2),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    bundle = store.get()
    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_loaded": bundle is not None,
            "model_version": store.version,
            "model_stage": MODEL_STAGE,
            "model_trained_on": bundle.get("last_date") if bundle else None,
        }
    )


@app.post("/predict")
def predict():
    bundle, err = require_model()
    if err:
        return err

    if request.data and not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    payload = request.get_json(silent=True) if request.data else {}
    if payload is None:
        raise BadRequest("Request body is not valid JSON")
    if not isinstance(payload, dict):
        raise BadRequest("Request body must be a JSON object")

    days = parse_forecast_days(payload)

    if "history" in payload:
        last_date, last_close, features = history_to_features(payload["history"])
        source = "custom_history"
    else:
        last_date = pd.Timestamp(bundle["last_date"])
        last_close = float(bundle["last_close"])
        features = bundle["last_features"]
        source = "latest_market_data"

    predictions = make_forecast(bundle, features, last_close, last_date, days)
    return jsonify(
        {
            "model_version": store.version,
            "input_source": source,
            "last_date": last_date.strftime("%Y-%m-%d"),
            "last_close": round(last_close, 2),
            "forecast_days": days,
            "predictions": predictions,
        }
    )


@app.get("/metrics")
def metrics():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return error_response(503, "Metrics not available", f"{path} not found")
    data = json.loads(path.read_text())
    keys = [
        "version",
        "mae",
        "rmse",
        "mape",
        "baseline_mape",
        "test_start",
        "test_end",
        "trained_at",
        "train_rows",
    ]
    return jsonify({k: data.get(k) for k in keys})


@app.post("/reload")
def reload_model():
    old = store.version
    if not store.load():
        return error_response(503, "Reload failed", store.error)
    return jsonify(
        {
            "status": "reloaded",
            "previous_version": old,
            "model_version": store.version,
            "loaded_at": store.loaded_at,
        }
    )


# ---------------------------------------------------------------------------
# Error handlers → always JSON
# ---------------------------------------------------------------------------
@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return error_response(400, "Bad Request", str(e))


@app.errorhandler(HTTPException)
def handle_http_error(e):
    return error_response(e.code, e.name, e.description)


@app.errorhandler(Exception)
def handle_unexpected(e):
    log.exception("Unhandled error")
    return error_response(
        500, "Internal Server Error", "unexpected error, see server logs"
    )


if __name__ == "__main__":  # dev server only — use gunicorn in production
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
