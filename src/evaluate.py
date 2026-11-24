"""
evaluate.py - Step 4: decide whether the new (candidate) model goes live.

Rule: promote the candidate only if its MAPE on the last 7 days
      is below MAPE_THRESHOLD (default 10%).

If promoted:   candidate.pkl -> model.pkl
               candidate_metrics.json -> metrics.json
               candidate_predictions.json -> predictions.json
If rejected:   the old live model stays. Nothing changes for Flask.
"""

import json
import os
import shutil

from src import config


def _replace(src, dst):
    """Copy to a temp file, then swap - so Flask never reads a half-written file."""
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")  # unique per process
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout)
        fout.flush()
        os.fsync(fout.fileno())  # data on disk before the rename (bind mounts)
    os.replace(tmp, dst)


def evaluate_model() -> dict:
    """Check candidate metrics against the threshold and promote if good. Returns the decision."""
    metrics = json.loads(config.CANDIDATE_METRICS.read_text())
    mape = metrics["mape"]
    passed = mape <= config.MAPE_THRESHOLD
    first_model = not config.LIVE_MODEL.exists()

    decision = {
        "version": metrics["version"],
        "mape": mape,
        "threshold": config.MAPE_THRESHOLD,
        "passed": passed,
        "promoted": False,
    }

    if passed:
        _replace(config.CANDIDATE_MODEL, config.LIVE_MODEL)
        _replace(config.CANDIDATE_PREDICTIONS, config.LIVE_PREDICTIONS)
        metrics["promoted"] = True
        config.LIVE_METRICS.write_text(json.dumps(metrics, indent=2))
        decision["promoted"] = True
        print(
            f"[evaluate] PROMOTED {metrics['version']} "
            f"(MAPE {mape:.2f}% <= {config.MAPE_THRESHOLD}%)"
        )
    else:
        print(
            f"[evaluate] REJECTED {metrics['version']} "
            f"(MAPE {mape:.2f}% > {config.MAPE_THRESHOLD}%). "
            + (
                "No live model exists yet!"
                if first_model
                else "Keeping the current live model."
            )
        )

    return decision


if __name__ == "__main__":
    evaluate_model()
