"""
preprocess.py — Step 2: clean raw JSON and add features.

Input:  data/raw.json
Output: data/clean.json

Cleaning:
  - sort by date, drop duplicate dates
  - drop rows with missing / zero / negative prices
  - fill small gaps (missing days) by carrying the previous value forward

Features (Part C1):
  - ma_7, ma_30        → 7-day and 30-day moving averages of close
  - volatility_7/_30   → std of daily returns over 7 / 30 days
  - return_1           → today's % change
  - lag_1 ... lag_7    → close price 1..7 days ago (as ratio to today's close)
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config

FEATURE_COLUMNS = [
    "return_1",
    "ma_7_ratio",
    "ma_30_ratio",
    "volatility_7",
    "volatility_30",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7",
    "hl_range",
    "volume_change",
]


def load_raw() -> pd.DataFrame:
    payload = json.loads(config.RAW_JSON.read_text())
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    price_cols = ["open", "high", "low", "close"]
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]

    # make sure every calendar day exists, fill gaps with previous day
    df = df.set_index("date").asfreq("D").ffill().reset_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators. Works on any DataFrame with date/open/high/low/close/volume."""
    df = df.copy()
    close = df["close"]

    df["return_1"] = close.pct_change()
    df["ma_7"] = close.rolling(7).mean()
    df["ma_30"] = close.rolling(30).mean()
    df["ma_7_ratio"] = df["ma_7"] / close
    df["ma_30_ratio"] = df["ma_30"] / close
    df["volatility_7"] = df["return_1"].rolling(7).std()
    df["volatility_30"] = df["return_1"].rolling(30).std()
    for lag in range(1, 8):
        df[f"lag_{lag}"] = close.shift(lag) / close  # ratio → scale-free
    df["hl_range"] = (df["high"] - df["low"]) / close
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0)

    # first ~30 rows don't have enough history for the features
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def preprocess_data() -> str:
    """Read raw.json, clean it, add features, save clean.json. Returns the file path."""
    config.ensure_dirs()
    raw = load_raw()
    print(f"[preprocess] Loaded {len(raw)} raw rows")

    df = add_features(clean(raw))

    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(out),
        "feature_columns": FEATURE_COLUMNS,
        "data": out.to_dict(orient="records"),
    }
    config.CLEAN_JSON.write_text(json.dumps(payload, indent=2))

    print(
        f"[preprocess] Saved {len(out)} clean rows with "
        f"{len(FEATURE_COLUMNS)} features to {config.CLEAN_JSON}"
    )
    return str(config.CLEAN_JSON)


if __name__ == "__main__":
    preprocess_data()
