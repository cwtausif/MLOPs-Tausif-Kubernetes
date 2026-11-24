"""
notify.py — Step 5: tell the Flask API to load the new model.

Sends:  POST {FLASK_RELOAD_URL}   (default http://localhost:5000/reload)

Only called when a new model was promoted.
If Flask isn't running yet, it prints a warning instead of failing
(set NOTIFY_STRICT=true to make it fail instead).
"""

import os

import requests

from src import config


def notify_flask(promoted: bool = True) -> bool:
    """Ask Flask to reload the model. Returns True if Flask confirmed."""
    if not promoted:
        print("[notify] No new model promoted → nothing to reload.")
        return False

    url = config.FLASK_RELOAD_URL
    try:
        r = requests.post(url, timeout=10)
        r.raise_for_status()
        print(
            f"[notify] ✅ Flask reloaded the model: {r.status_code} {r.text.strip()[:200]}"
        )
        return True
    except requests.RequestException as e:
        msg = f"[notify] ⚠️ Could not reach Flask at {url}: {e}"
        if os.getenv("NOTIFY_STRICT", "false").lower() == "true":
            raise RuntimeError(msg) from e
        print(msg + " (skipping — is the Flask app running?)")
        return False


if __name__ == "__main__":
    notify_flask()
