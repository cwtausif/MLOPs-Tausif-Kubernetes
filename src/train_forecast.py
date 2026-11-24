"""
train_forecast.py — Step 3: train the 7-day BTC forecasting model.

Input:  data/clean.json
Output: models/candidate.pkl              (new model — NOT live yet)
        models/candidate_metrics.json     (MAE, RMSE, MAPE on the last 7 days)
        models/candidate_predictions.json (forecast for the next 7 days)

How the model works ("direct" multi-step forecasting):
  - 7 small XGBoost models, one per day ahead (day+1 ... day+7).
  - Each predicts the % change from today's close to that day's close.
    (Predicting % change instead of raw price works much better for trees,
     because BTC can reach prices it has never seen before.)

How it is tested (honest evaluation):
  - Hide the last 7 days, train on everything before, predict those 7 days,
    compare with the real prices → MAE, RMSE, MAPE.
  - Then retrain on ALL data and forecast the real next 7 days.
"""

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src import config
from src.preprocess import FEATURE_COLUMNS

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def load_clean() -> pd.DataFrame:
    payload = json.loads(config.CLEAN_JSON.read_text())
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fit_models(df: pd.DataFrame, last_index: int) -> dict:
    """
    Train one model per horizon using only data up to row `last_index`
    (targets must not go past last_index — no peeking into the future).
    """
    log_close = np.log(df["close"].to_numpy())
    models = {}
    for h in range(1, config.FORECAST_DAYS + 1):
        rows = np.arange(0, last_index - h + 1)  # rows whose target is known
        X = df.loc[rows, FEATURE_COLUMNS].to_numpy()
        y = log_close[rows + h] - log_close[rows]  # log return h days ahead
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X, y)
        models[h] = model
    return models


def forecast(models: dict, features_row: np.ndarray, last_close: float) -> list:
    """Turn the 7 predicted % changes into 7 prices."""
    x = features_row.reshape(1, -1)
    return [
        float(last_close * np.exp(models[h].predict(x)[0]))
        for h in range(1, config.FORECAST_DAYS + 1)
    ]


def evaluate_on_last_week(df: pd.DataFrame) -> dict:
    """Hide the last 7 days, train on the rest, predict those 7 days, score."""
    n = len(df)
    origin = n - 1 - config.FORECAST_DAYS  # "today" in the test
    models = fit_models(df, origin)

    preds = forecast(
        models, df.loc[origin, FEATURE_COLUMNS].to_numpy(), df.loc[origin, "close"]
    )
    actual = df.loc[origin + 1 : origin + config.FORECAST_DAYS, "close"].to_numpy()
    naive = np.full_like(
        actual, df.loc[origin, "close"]
    )  # baseline: "price stays the same"

    def mape(a, p):
        return float(np.mean(np.abs((a - p) / a)) * 100)

    return {
        "test_start": df.loc[origin + 1, "date"].strftime("%Y-%m-%d"),
        "test_end": df.loc[origin + config.FORECAST_DAYS, "date"].strftime("%Y-%m-%d"),
        "mae": float(mean_absolute_error(actual, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
        "mape": mape(actual, preds),
        "baseline_mape": mape(actual, naive),
        "test_actual": [float(v) for v in actual],
        "test_predicted": preds,
    }


def train_model() -> dict:
    """Evaluate, retrain on all data, forecast next 7 days, save candidate files."""
    config.ensure_dirs()
    df = load_clean()
    print(
        f"[train] Loaded {len(df)} rows "
        f"({df['date'].iloc[0]:%Y-%m-%d} → {df['date'].iloc[-1]:%Y-%m-%d})"
    )

    # 1) honest test on the last 7 days
    metrics = evaluate_on_last_week(df)
    print(
        f"[train] Test {metrics['test_start']} → {metrics['test_end']}: "
        f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
        f"MAPE={metrics['mape']:.2f}%  (baseline MAPE={metrics['baseline_mape']:.2f}%)"
    )

    # 2) final model on ALL data
    last = len(df) - 1
    models = fit_models(df, last)
    last_date = df.loc[last, "date"]
    last_close = float(df.loc[last, "close"])
    last_features = df.loc[last, FEATURE_COLUMNS].to_numpy(dtype=float)
    prices = forecast(models, last_features, last_close)
    dates = [
        (last_date + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        for h in range(1, config.FORECAST_DAYS + 1)
    ]

    now = datetime.now(timezone.utc)
    version = now.strftime("v%Y%m%d-%H%M%S")

    # 3) save candidate files
    bundle = {
        "version": version,
        "trained_at": now.isoformat(),
        "models": models,
        "feature_columns": FEATURE_COLUMNS,
        "forecast_days": config.FORECAST_DAYS,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "last_close": last_close,
        "last_features": last_features,
    }
    with open(config.CANDIDATE_MODEL, "wb") as f:
        pickle.dump(bundle, f)

    metrics.update(
        {"version": version, "trained_at": now.isoformat(), "train_rows": len(df)}
    )
    config.CANDIDATE_METRICS.write_text(json.dumps(metrics, indent=2))

    predictions = {
        "version": version,
        "generated_at": now.isoformat(),
        "last_date": bundle["last_date"],
        "last_close": last_close,
        "predictions": [
            {"date": d, "close": round(p, 2)} for d, p in zip(dates, prices)
        ],
    }
    config.CANDIDATE_PREDICTIONS.write_text(json.dumps(predictions, indent=2))

    print(
        f"[train] Saved candidate {version}. Next 7 days: "
        + ", ".join(f"{d}={p:,.0f}" for d, p in zip(dates, prices))
    )
    return {"version": version, "mape": metrics["mape"]}


if __name__ == "__main__":
    train_model()
