"""
etl_fetch.py — Step 1: download daily BTC-USD prices and save them as raw JSON.

Output: data/raw.json
  [{"date": "2026-11-16", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
"""

import json
from datetime import datetime, timezone

import yfinance as yf

from src import config


def fetch_data() -> str:
    """Download BTC prices from Yahoo Finance and save to raw.json. Returns the file path."""
    config.ensure_dirs()
    print(f"[fetch] Downloading {config.TICKER} from {config.START_DATE} ...")

    df = yf.Ticker(config.TICKER).history(start=config.START_DATE, interval="1d")
    if df.empty:
        raise RuntimeError(f"No data returned for {config.TICKER}")

    records = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for idx, row in df.iterrows()
    ]

    payload = {
        "ticker": config.TICKER,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "data": records,
    }
    config.RAW_JSON.write_text(json.dumps(payload, indent=2))

    print(
        f"[fetch] Saved {len(records)} rows "
        f"({records[0]['date']} → {records[-1]['date']}) to {config.RAW_JSON}"
    )
    return str(config.RAW_JSON)


if __name__ == "__main__":
    fetch_data()
