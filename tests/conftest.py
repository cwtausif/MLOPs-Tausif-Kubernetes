"""
Shared test setup.

- Makes `import app` (app/app.py) and `import src...` work.
- Builds a tiny fake model in a temp folder, so tests don't need the real
  trained model or internet access.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from src.preprocess import FEATURE_COLUMNS  # noqa: E402


def make_bundle(version="vTEST"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_COLUMNS)))
    models = {}
    for h in range(1, 8):
        m = XGBRegressor(n_estimators=5, max_depth=2)
        m.fit(X, rng.normal(0, 0.01, size=60))
        models[h] = m
    return {
        "version": version,
        "models": models,
        "feature_columns": FEATURE_COLUMNS,
        "forecast_days": 7,
        "last_date": "2026-11-17",
        "last_close": 65000.0,
        "last_features": X[-1],
    }


def write_model(model_dir: Path, version="vTEST"):
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(make_bundle(version), f)


@pytest.fixture
def model_dir(tmp_path):
    write_model(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "version": "vTEST",
                "mae": 1000.0,
                "rmse": 1200.0,
                "mape": 2.5,
                "baseline_mape": 3.0,
                "test_start": "2026-11-11",
                "test_end": "2026-11-17",
                "trained_at": "2026-11-17T00:00:00+00:00",
                "train_rows": 3500,
            }
        )
    )
    return tmp_path


@pytest.fixture
def client(model_dir, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
    app_module.store.__init__()  # fresh store pointing at the temp model
    app_module.store.load()
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def empty_client(tmp_path, monkeypatch):
    """A client with NO model on disk."""
    import app as app_module

    monkeypatch.setattr(app_module, "MODEL_DIR", tmp_path)
    app_module.store.__init__()
    app_module.store.load()
    return app_module.app.test_client()


@pytest.fixture
def history():
    """60 days of valid OHLCV records."""
    rng = np.random.default_rng(1)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
    return [
        {
            "date": f"2026-{9 + (i // 30):02d}-{(i % 30) + 1:02d}",
            "open": float(c * 0.99),
            "high": float(c * 1.02),
            "low": float(c * 0.97),
            "close": float(c),
            "volume": float(1e10 + i),
        }
        for i, c in enumerate(close)
    ]
