"""Unit tests for the Flask API (Part A3). Run:  pytest -v"""

import os
import time

from conftest import write_model


# ---------- /health ----------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "vTEST"
    assert body["uptime_seconds"] >= 0


def test_health_without_model_still_alive(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_loaded"] is False


# ---------- /predict ----------
def test_predict_default_7_days(client):
    r = client.post("/predict")
    assert r.status_code == 200
    body = r.get_json()
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    assert body["predictions"][0]["date"] == "2026-11-18"
    assert body["predictions"][-1]["date"] == "2026-11-24"
    assert all(p["close"] > 0 for p in body["predictions"])
    assert body["input_source"] == "latest_market_data"


def test_predict_custom_days(client):
    r = client.post("/predict", json={"forecast_days": 3})
    assert r.status_code == 200
    assert len(r.get_json()["predictions"]) == 3


def test_predict_with_custom_history(client, history):
    r = client.post("/predict", json={"forecast_days": 7, "history": history})
    assert r.status_code == 200
    body = r.get_json()
    assert body["input_source"] == "custom_history"
    assert body["last_date"] == "2026-10-30"
    assert len(body["predictions"]) == 7


def test_predict_rejects_bad_forecast_days(client):
    for bad in [0, 8, -1, "seven", 3.5, True, None]:
        r = client.post("/predict", json={"forecast_days": bad})
        assert r.status_code == 400, bad
        assert r.get_json()["error"] == "Bad Request"


def test_predict_rejects_invalid_json(client):
    r = client.post("/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400


def test_predict_rejects_non_json_body(client):
    r = client.post("/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_predict_rejects_json_list(client):
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 400


def test_predict_rejects_short_history(client, history):
    r = client.post("/predict", json={"history": history[:10]})
    assert r.status_code == 400
    assert "at least" in r.get_json()["detail"]


def test_predict_rejects_history_missing_field(client, history):
    del history[5]["close"]
    r = client.post("/predict", json={"history": history})
    assert r.status_code == 400
    assert "close" in r.get_json()["detail"]


def test_predict_rejects_negative_price(client, history):
    history[3]["low"] = -5
    r = client.post("/predict", json={"history": history})
    assert r.status_code == 400


def test_predict_without_model_returns_503(empty_client):
    r = empty_client.post("/predict")
    assert r.status_code == 503


# ---------- /metrics ----------
def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.get_json()
    for key in ["mae", "rmse", "mape", "version"]:
        assert key in body
    assert body["mape"] == 2.5


def test_metrics_missing_returns_503(empty_client):
    assert empty_client.get("/metrics").status_code == 503


# ---------- /reload ----------
def test_reload_picks_up_new_model(client, model_dir):
    write_model(model_dir, version="vNEW")
    r = client.post("/reload")
    assert r.status_code == 200
    body = r.get_json()
    assert body["previous_version"] == "vTEST"
    assert body["model_version"] == "vNEW"


def test_auto_reload_when_file_changes(client, model_dir):
    """Without calling /reload, a newer model.pkl is picked up automatically."""
    write_model(model_dir, version="vAUTO")
    later = time.time() + 5
    os.utime(model_dir / "model.pkl", (later, later))
    assert client.get("/health").get_json()["model_version"] == "vAUTO"


def test_reload_without_model_returns_503(empty_client):
    assert empty_client.post("/reload").status_code == 503


# ---------- errors ----------
def test_unknown_route_returns_json_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert r.get_json()["status"] == 404


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_internal_error_returns_500(client, monkeypatch):
    import app as app_module

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "make_forecast", boom)
    r = client.post("/predict")
    assert r.status_code == 500
    assert r.get_json()["error"] == "Internal Server Error"
