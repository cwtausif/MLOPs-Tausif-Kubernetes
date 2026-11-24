# BTC-USD 7-Day Forecast — Production MLOps Pipeline

End-to-end MLOps system that ingests daily Bitcoin prices, retrains an XGBoost
forecaster every day, and serves 7-day close-price forecasts from an
auto-scaling Flask microservice on Kubernetes.

```
yfinance ─► Airflow DAG (fetch → features → train → evaluate → reload) ─► models/ (PVC)
                                                                               │
GitHub push ─► Actions (flake8, black, pytest, KFP compile) ─► GHCR / Docker Hub image
                                                                               │
                     Ingress btc-forecast.local ─► Service ─► Deployment (3 pods, HPA 2–8)
```

| Path | Purpose |
|---|---|
| `app/app.py` | Flask API: `GET /health`, `POST /predict`, `GET /metrics`, `POST /reload` |
| `app/Dockerfile` | Multi-stage, non-root, `python:3.12-slim`, Gunicorn |
| `src/` | ETL (`etl_fetch.py`), features (`preprocess.py`), training (`train_forecast.py`), promotion gate (`evaluate.py`), reload trigger (`notify.py`) |
| `dags/btc_etl_pipeline.py` | Daily Airflow DAG |
| `kubeflow/btc_kfp_pipeline.py` | KFP v2 container-component pipeline → `btc_kfp_pipeline.yaml` |
| `.github/workflows/ci_cd.yml` | Lint, test, build, push (SHA + `latest`) |
| `k8s/` | ConfigMap/Secret, PVC, Deployment, Services, HPA, Ingress |
| `loadtest/` | Locust file + in-cluster Locust Job for the HPA demo |
| `docs/` | Technical report (PDF), CLI evidence and screenshots |

## Run it

```bash
./setup.sh                                   # tools + .venv + minikube (macOS)
source .venv/bin/activate && pytest -v       # 25 unit tests
docker compose up -d --build                 # Flask :5001, Airflow :8080 (airflow/airflow)
docker compose exec airflow-scheduler airflow dags trigger btc_etl_pipeline

minikube image load ghcr.io/cwtausif/btc-forecast-api:latest   # or pull from GHCR
kubectl apply -f k8s/configmap-secret.yaml -f k8s/pvc.yaml -f k8s/model-loader.yaml
./k8s/load_model.sh                          # copy trained model into the PVC
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml -f k8s/ingress.yaml

curl -X POST -H 'Host: btc-forecast.local' http://$(minikube ip)/predict \
     -H 'Content-Type: application/json' -d '{"forecast_days": 7}'
```

Kubeflow: `python kubeflow/btc_kfp_pipeline.py` compiles the spec; `python kubeflow/run_local.py` runs every compiled step in Docker.

Load test: `kubectl create configmap locustfile --from-file=loadtest/locustfile.py && kubectl apply -f loadtest/locust-job.yaml`, then `kubectl get hpa --watch`.

CI pushes to `ghcr.io/cwtausif/btc-forecast-api` with the built-in token; add the
`DOCKER_USERNAME` / `DOCKER_PASSWORD` repository secrets to also push to Docker Hub.
