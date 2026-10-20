"""Send random patients to the API so the dashboards have something to show.

Run:  python scripts/load_test.py http://localhost:8000 300
"""
import random
import sys
import time

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def random_patient():
    return {"age": random.randint(30, 77), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
            "trestbps": random.randint(94, 200), "chol": random.randint(126, 560),
            "fbs": random.randint(0, 1), "restecg": random.randint(0, 2),
            "thalach": random.randint(71, 202), "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 6), 1), "slope": random.randint(1, 3),
            "ca": random.randint(0, 3), "thal": random.choice([3, 6, 7])}


ok = errors = 0
for i in range(1, count + 1):
    patient = random_patient()
    if i % 15 == 0:
        patient["age"] = -1              # every 15th request is invalid on purpose -> shows up as an error
    status = requests.post(f"{url}/predict", json=patient, timeout=10).status_code
    ok += status == 200
    errors += status != 200
    time.sleep(0.05)
print(f"Sent {count} requests: {ok} OK, {errors} rejected")
