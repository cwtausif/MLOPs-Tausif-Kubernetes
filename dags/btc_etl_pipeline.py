"""
btc_etl_pipeline.py — Airflow DAG: runs the BTC pipeline every day.

    fetch_data → preprocess → train_model → evaluate_model → reload_api

Each task just calls a function from src/ — the real work lives there.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

default_args = {
    "owner": "mlops",
    "retries": 2,  # retry failed tasks (e.g. network errors)
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD fetch → features → retrain → evaluate → reload Flask",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,  # don't run for past missed days
    default_args=default_args,
    tags=["btc", "mlops"],
)
def btc_etl_pipeline():

    @task
    def fetch_data() -> str:
        from src.etl_fetch import fetch_data as run

        return run()

    @task
    def preprocess(_raw_path: str) -> str:
        from src.preprocess import preprocess_data as run

        return run()

    @task
    def train_model(_clean_path: str) -> dict:
        from src.train_forecast import train_model as run

        return run()

    @task
    def evaluate_model(_train_result: dict) -> dict:
        from src.evaluate import evaluate_model as run

        return run()

    @task
    def reload_api(decision: dict) -> bool:
        from src.notify import notify_flask as run

        return run(promoted=decision["promoted"])

    # the order: each task gets the previous task's output → Airflow draws the arrows
    raw = fetch_data()
    clean = preprocess(raw)
    trained = train_model(clean)
    decision = evaluate_model(trained)
    reload_api(decision)


btc_etl_pipeline()
