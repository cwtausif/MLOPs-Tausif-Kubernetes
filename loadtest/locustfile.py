"""
Locust load test for the HPA demo (Part D5).

Start (web UI with live graphs at http://localhost:8089):
    locust -f loadtest/locustfile.py --host http://btc-forecast.local

Or headless for 3 minutes with 200 users:
    locust -f loadtest/locustfile.py --host http://btc-forecast.local \
        --headless -u 200 -r 20 -t 3m

While it runs, watch the pods scale in another terminal:
    kubectl get hpa --watch
"""

from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        # the heaviest endpoint -> drives CPU up
        self.client.post("/predict", json={"forecast_days": 7})

    @task(2)
    def predict_short(self):
        self.client.post(
            "/predict", json={"forecast_days": 3}, name="/predict (3 days)"
        )

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
