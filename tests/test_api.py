"""Tests for the FastAPI service (app/main.py). The tests plug a small test model into the app."""
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app import main
from src.data import clean, split_xy
from src.features import build_model_pipeline


@pytest.fixture
def client(monkeypatch, raw_df):
    X, y = split_xy(clean(raw_df))
    test_model = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    monkeypatch.setattr(main, "model", test_model)            # give the app a model for this test
    monkeypatch.setattr(main, "model_name", "test-logreg")
    return TestClient(main.app)


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_fails_without_model(client, monkeypatch):
    monkeypatch.setattr(main, "model", None)
    assert client.get("/health").status_code == 503


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    response = client.post("/predict", json=sample_patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test-logreg"


def test_invalid_input_is_rejected(client, sample_patient):
    assert client.post("/predict", json=dict(sample_patient, age=-5)).status_code == 422
    missing_cholesterol = {k: v for k, v in sample_patient.items() if k != "chol"}
    assert client.post("/predict", json=missing_cholesterol).status_code == 422


def test_metrics_page_counts_requests(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
