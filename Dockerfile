# ---------- Heart Disease Risk API ----------
FROM python:3.11-slim

# print logs immediately (so `docker logs` shows them)
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1) install libraries first (Docker caches this step, so rebuilds are fast)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) copy our code and the trained model
COPY src/ ./src/
COPY app/ ./app/
COPY models/model.joblib models/metadata.json ./models/

# 3) do not run as root (good security practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
