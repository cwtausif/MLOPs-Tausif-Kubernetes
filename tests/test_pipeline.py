"""Unit tests for the ETL / training / promotion code in src/ (no internet needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, evaluate, preprocess, train_forecast
from src.preprocess import FEATURE_COLUMNS, add_features, clean


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every config path at a temp folder and write a synthetic raw.json."""
    data, models = tmp_path / "data", tmp_path / "models"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "MODEL_DIR", models)
    for name, folder in [("RAW_JSON", data), ("CLEAN_JSON", data)]:
        monkeypatch.setattr(config, name, folder / getattr(config, name).name)
    for name in [
        "CANDIDATE_MODEL",
        "CANDIDATE_METRICS",
        "CANDIDATE_PREDICTIONS",
        "LIVE_MODEL",
        "LIVE_METRICS",
        "LIVE_PREDICTIONS",
    ]:
        monkeypatch.setattr(config, name, models / getattr(config, name).name)
    config.ensure_dirs()

    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=220, freq="D")
    close = 50000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "open": c * 0.995,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e10 + i,
        }
        for i, (d, c) in enumerate(zip(dates, close))
    ]
    config.RAW_JSON.write_text(json.dumps({"data": rows}))
    return tmp_path


def test_clean_fills_missing_days_and_drops_bad_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04"]),
            "open": [1.0, 2.0, -1.0],
            "high": [1.0, 2.0, 1.0],
            "low": [1.0, 2.0, 1.0],
            "close": [1.0, 2.0, 1.0],
            "volume": [1.0, 2.0, 1.0],
        }
    )
    out = clean(df)
    assert list(out["date"].dt.day) == [1, 2, 3]  # gap filled, bad row dropped
    assert out.loc[1, "close"] == 1.0  # forward-filled


def test_add_features_has_all_columns_and_no_nans(workspace):
    df = preprocess.load_raw()
    feats = add_features(df)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert not feats[FEATURE_COLUMNS].isna().any().any()
    assert len(feats) == len(df) - 30  # first 30 rows lack 30-day history


def test_full_pipeline_trains_and_promotes(workspace, monkeypatch):
    monkeypatch.setattr(config, "MAPE_THRESHOLD", 100.0)
    preprocess.preprocess_data()
    result = train_forecast.train_model()
    assert result["mape"] >= 0

    metrics = json.loads(config.CANDIDATE_METRICS.read_text())
    for key in ["mae", "rmse", "mape", "baseline_mape", "test_start", "test_end"]:
        assert key in metrics
    preds = json.loads(config.CANDIDATE_PREDICTIONS.read_text())["predictions"]
    assert len(preds) == 7 and all(p["close"] > 0 for p in preds)

    decision = evaluate.evaluate_model()
    assert decision["promoted"] is True
    assert config.LIVE_MODEL.exists() and config.LIVE_METRICS.exists()


def test_evaluate_rejects_model_above_threshold(workspace, monkeypatch):
    monkeypatch.setattr(config, "MAPE_THRESHOLD", 0.0)
    config.CANDIDATE_METRICS.write_text(json.dumps({"version": "vBAD", "mape": 9.9}))
    decision = evaluate.evaluate_model()
    assert decision["promoted"] is False
    assert not config.LIVE_MODEL.exists()
