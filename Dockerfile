# DR-Sight ML API — CPU inference image.
# EfficientNet-B3 on CPU is a few seconds per image: fine for a demo. If it is
# noticeably slow during judging, deploy an "edge" (MobileNetV3-Small) checkpoint
# instead (set DR_BACKBONE=edge and mount matching weights) — see DEPLOY.md.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# libgomp1 is needed by the PyTorch CPU runtime. opencv-python-headless needs no
# system GL/GLib libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for layer caching. Pull the CPU-only torch wheels so the
# image stays ~1GB instead of ~3GB (default PyPI torch bundles CUDA).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# App code + trained weights.
COPY src/ ./src/
# Ship a checkpoint at models/best.pt (real weights, or the edge variant).
# Kept optional so the image builds in CI without weights; /health reports
# model_loaded=false until one is present / mounted.
COPY models/ ./models/

EXPOSE 8000

# Required at runtime (fail closed if unset):
#   API_KEY          shared secret the Base44 backend function sends
#   ALLOWED_ORIGIN   e.g. https://your-app.base44.app  (comma-separated, never *)
# Optional:
#   DR_BACKBONE=b3|edge   DR_WEIGHTS=models/best.pt   DR_DEVICE=cpu
#   DR_UNCERTAINTY_THRESHOLD=0.6   MAX_UPLOAD_BYTES=12582912
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
