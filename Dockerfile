# KinWhy AI inference server: POST /api/analyze-image for the mobile app and the wearable camera.
# deploy/cloudrun.sh stages the chosen model as deploy/model/model_fp32.onnx + model.json, then deploys.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FOOD_MODEL_ONNX=/app/model/model_fp32.onnx \
    FOOD_MODEL_META=/app/model/model.json \
    PRELOAD_MODEL=1 \
    SERVE_STATIC=0 \
    PORT=8080

WORKDIR /app
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY backend/ backend/
COPY deploy/model/model_fp32.onnx deploy/model/model.json model/

RUN useradd --system --uid 10001 app && chown -R app /app
USER app

# One worker per instance: the model holds ~0.5 GB and inference is serialised per process;
# Cloud Run scales by adding instances.
CMD exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips="*"
