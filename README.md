# MLOps Assignment 3: Bitcoin Price Forecast

Tausif (22L-8072)

This project predicts the Bitcoin (BTC-USD) closing price for the next 7 days.
Airflow downloads the data every day and trains an XGBoost model. A Flask API serves
the predictions and runs on Kubernetes (Minikube). GitHub Actions tests the code and
pushes the Docker image.

The report is in `docs/report.pdf`.

## Folders

- `app/` Flask API and Dockerfile
- `src/` code to fetch data, make features, train and evaluate the model
- `dags/` Airflow DAG
- `kubeflow/` Kubeflow pipeline
- `k8s/` Kubernetes files
- `loadtest/` Locust load test
- `tests/` pytest tests
- `docs/` report and screenshots

## How to run

Install everything (Mac):

```bash
./setup.sh
source .venv/bin/activate
```

Run the tests:

```bash
pytest -v
```

Start Flask, Airflow and Postgres with Docker Compose:

```bash
docker compose up -d --build
```

Airflow is at http://localhost:8080 (user `airflow`, password `airflow`) and Flask is at http://localhost:5001.

Deploy on Minikube:

```bash
kubectl apply -f k8s/configmap-secret.yaml -f k8s/pvc.yaml -f k8s/model-loader.yaml
./k8s/load_model.sh
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml -f k8s/ingress.yaml
```

Get a forecast:

```bash
curl -X POST http://btc-forecast.local/predict -H "Content-Type: application/json" -d '{"forecast_days": 7}'
```

Load test:

```bash
kubectl create configmap locustfile --from-file=loadtest/locustfile.py
kubectl apply -f loadtest/locust-job.yaml
kubectl get hpa --watch
```

Kubeflow:

```bash
python kubeflow/btc_kfp_pipeline.py
python kubeflow/run_local.py
```

## Docker image

`ghcr.io/cwtausif/btc-forecast-api:latest`

To also push to Docker Hub, add `DOCKER_USERNAME` and `DOCKER_PASSWORD` in the repo secrets.
