"""FastAPI service that serves the heart-disease model.

  GET  /health   -> is the service (and the model) OK?
  POST /predict  -> patient values as JSON in, prediction + confidence out
  GET  /metrics  -> numbers for Prometheus (request count, latency, predictions per class)
  GET  /docs     -> automatic test page (Swagger UI)

Run locally:  uvicorn app.main:app --port 8000
"""
import json
import logging
import os
import sys
import time
import uuid

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.predict import predict_records

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")

# ---------- 1) logging: one JSON line per event, printed to the console ----------
logger = logging.getLogger("heart-api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

# ---------- 2) metrics that Prometheus collects ----------
REQUESTS = Counter("api_requests_total", "Number of API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Time to answer a request (seconds)", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions made, per class", ["prediction"])

# ---------- 3) load the model once, when the service starts ----------
model = None
model_name = "unknown"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        model_name = json.load(f).get("model_name", "unknown")


class PatientData(BaseModel):
    """The 13 inputs. Values outside the allowed range are rejected automatically (HTTP 422)."""
    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}}}

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Highest heart rate reached")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of the ST segment (1-3)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=3, le=7, description="3 = normal, 6 = fixed defect, 7 = reversible defect")


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}         # allow a field called model_name
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    model_name: str


app = FastAPI(title="Heart Disease Risk API", version="1.0.0")


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    """Runs around EVERY request: measures the time, counts it, and writes one log line."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        seconds = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(seconds)
            logger.info(json.dumps({"event": "request", "id": uuid.uuid4().hex[:8], "method": request.method,
                                    "path": path, "status": status, "ms": round(seconds * 1000, 2)}))


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": model_name}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predict_records([patient.model_dump()], model)[0]
    PREDICTIONS.labels(str(result["prediction"])).inc()
    logger.info(json.dumps({"event": "prediction", "input": patient.model_dump(), "output": result}))
    return PredictionResponse(**result, model_name=model_name)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
