"""
btc_kfp_pipeline.py - Kubeflow Pipelines (KFP v2) version of the BTC pipeline (Part C2).

Four discrete *container components*, each running in the `btc-pipeline` image
(kubeflow/Dockerfile = python:3.12-slim + src/ + ML libraries):

    ingest_data -> generate_features -> train_model -> evaluate_and_register
      raw.json         clean.json          model.pkl         promoted model +
                                            metrics.json      registration record

Artifacts are passed between steps by Kubeflow (Dataset / Model / Metrics),
so every run is reproducible and its outputs are tracked in the KFP UI.

Compile to a YAML spec (no cluster needed):
    python kubeflow/btc_kfp_pipeline.py
    -> kubeflow/btc_kfp_pipeline.yaml

Upload + schedule daily on a Kubeflow cluster (optional):
    KFP_HOST=http://localhost:8080 python kubeflow/btc_kfp_pipeline.py --schedule
"""

import argparse
import os
from pathlib import Path

from kfp import compiler, dsl

IMAGE = os.getenv("KFP_IMAGE", "ghcr.io/cwtausif/btc-pipeline:latest")
HERE = Path(__file__).resolve().parent
PIPELINE_YAML = HERE / "btc_kfp_pipeline.yaml"

# Each step runs our src/ code with DATA_DIR / MODEL_DIR pointed at a scratch
# folder, then copies the result to the Kubeflow-managed artifact path.


@dsl.container_component
def ingest_data(ticker: str, start_date: str, raw_data: dsl.Output[dsl.Dataset]):
    """Download daily BTC-USD OHLCV data with yfinance -> raw.json."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            'set -e; export TICKER="$0" START_DATE="$1" DATA_DIR=/tmp/data MODEL_DIR=/tmp/models; '
            "python -m src.etl_fetch; "
            'mkdir -p "$(dirname "$2")"; cp /tmp/data/raw.json "$2"',
            ticker,
            start_date,
            raw_data.path,
        ],
    )


@dsl.container_component
def generate_features(
    raw_data: dsl.Input[dsl.Dataset], clean_data: dsl.Output[dsl.Dataset]
):
    """Clean data + technical indicators (MA7/MA30, volatility, lags) -> clean.json."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            "set -e; export DATA_DIR=/tmp/data MODEL_DIR=/tmp/models; mkdir -p $DATA_DIR; "
            'cp "$0" $DATA_DIR/raw.json; python -m src.preprocess; '
            'mkdir -p "$(dirname "$1")"; cp $DATA_DIR/clean.json "$1"',
            raw_data.path,
            clean_data.path,
        ],
    )


@dsl.container_component
def train_model(
    clean_data: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Artifact],
):
    """Train 7 XGBoost horizon models, score on the last 7 days."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            "set -e; export DATA_DIR=/tmp/data MODEL_DIR=/tmp/models; "
            "mkdir -p $DATA_DIR $MODEL_DIR; "
            'cp "$0" $DATA_DIR/clean.json; python -m src.train_forecast; '
            'mkdir -p "$(dirname "$1")" "$(dirname "$2")"; '
            'cp $MODEL_DIR/candidate.pkl "$1"; '
            'cp $MODEL_DIR/candidate_metrics.json "$2"',
            clean_data.path,
            model.path,
            metrics.path,
        ],
    )


@dsl.container_component
def evaluate_and_register(
    model: dsl.Input[dsl.Model],
    metrics: dsl.Input[dsl.Artifact],
    mape_threshold: float,
    registered_model: dsl.Output[dsl.Model],
):
    """Promote the model only if MAPE <= threshold (fails the run otherwise)."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c"],
        args=[
            'set -e; export MODEL_DIR=/tmp/models MAPE_THRESHOLD="$2"; '
            "mkdir -p $MODEL_DIR; "
            'cp "$0" $MODEL_DIR/candidate.pkl; '
            'cp "$1" $MODEL_DIR/candidate_metrics.json; '
            "echo '{}' > $MODEL_DIR/candidate_predictions.json; "
            "python -m src.evaluate; "
            "test -f $MODEL_DIR/model.pkl || "
            "{ echo 'model rejected: MAPE above threshold'; exit 1; }; "
            'mkdir -p "$(dirname "$3")"; cp $MODEL_DIR/model.pkl "$3"',
            model.path,
            metrics.path,
            mape_threshold,
            registered_model.path,
        ],
    )


@dsl.pipeline(
    name="btc-7day-forecast",
    description="Daily BTC-USD ingest -> features -> XGBoost 7-day forecast -> gate",
)
def btc_forecast_pipeline(
    ticker: str = "BTC-USD",
    start_date: str = "2017-01-01",
    mape_threshold: float = 10.0,
):
    ingest = ingest_data(ticker=ticker, start_date=start_date)

    features = generate_features(raw_data=ingest.outputs["raw_data"])

    train = train_model(clean_data=features.outputs["clean_data"])
    train.set_cpu_limit("2").set_memory_limit("2G")

    gate = evaluate_and_register(
        model=train.outputs["model"],
        metrics=train.outputs["metrics"],
        mape_threshold=mape_threshold,
    )

    for task in (ingest, features, train, gate):
        task.set_caching_options(False)  # always fetch today's data


def compile_pipeline() -> Path:
    compiler.Compiler().compile(btc_forecast_pipeline, str(PIPELINE_YAML))
    print(f"Compiled pipeline -> {PIPELINE_YAML}")
    return PIPELINE_YAML


def schedule_daily(host: str):
    """Upload the compiled spec and create a daily recurring run (needs a KFP cluster)."""
    import kfp

    client = kfp.Client(host=host)
    exp = client.create_experiment("btc-forecast")
    run = client.create_recurring_run(
        experiment_id=exp.experiment_id,
        job_name="btc-daily-retrain",
        cron_expression="0 0 2 * * *",  # every day 02:00 UTC
        pipeline_package_path=str(PIPELINE_YAML),
    )
    print(f"Recurring run created: {run.recurring_run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true", help="upload + run daily")
    args = parser.parse_args()
    compile_pipeline()
    if args.schedule:
        schedule_daily(os.environ.get("KFP_HOST", "http://localhost:8080"))
