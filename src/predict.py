"""Make predictions with the saved model (used by the API and from the command line).

Run:  python -m src.predict scripts/sample_request.json
"""
import json
import sys

import joblib
import pandas as pd

from src import config


def load_model(path=config.MODEL_FILE):
    """Load the saved pipeline."""
    return joblib.load(path)


def predict_records(records, model):
    """records = list of dicts with the 13 patient values. Returns one result dict per patient."""
    df = pd.DataFrame(records)[config.FEATURES].astype(float)
    probabilities = model.predict_proba(df)[:, 1]            # chance of heart disease, 0..1
    results = []
    for p in probabilities:
        sick = int(p >= 0.5)
        results.append({
            "prediction": sick,
            "label": "Heart disease risk" if sick else "No heart disease risk",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if sick else 1 - p), 4),   # how sure the model is of its answer
        })
    return results


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        patient = json.load(f)
    print(json.dumps(predict_records([patient], load_model()), indent=2))
