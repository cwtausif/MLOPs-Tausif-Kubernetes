"""
main.py — run the whole pipeline once, in order (no Airflow needed).

    fetch → preprocess → train → evaluate → notify

Run inside the Airflow container (it has all the libraries):
    docker compose run --rm airflow-scheduler python -m src.main
"""

from src.etl_fetch import fetch_data
from src.evaluate import evaluate_model
from src.notify import notify_flask
from src.preprocess import preprocess_data
from src.train_forecast import train_model


def run_pipeline():
    print("=" * 60)
    print(" BTC forecasting pipeline")
    print("=" * 60)
    fetch_data()
    preprocess_data()
    train_model()
    decision = evaluate_model()
    notify_flask(promoted=decision["promoted"])
    print("=" * 60)
    print(
        " Done:", "new model is LIVE" if decision["promoted"] else "kept previous model"
    )
    print("=" * 60)
    return decision


if __name__ == "__main__":
    run_pipeline()
