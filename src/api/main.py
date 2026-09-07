"""FastAPI service for DR-Sight.

Endpoints
  GET  /health   - liveness + whether real trained weights are loaded
  POST /predict  - screen one fundus image, return grade + Grad-CAM + referral

Auth
  Every /predict call must send `X-API-Key: <key>` matching the API_KEY env
  var. This is the only thing between the public internet and the model once
  deployed, and it is also what the Base44 Backend Function sends.

CORS
  Only the origin(s) in ALLOWED_ORIGIN (comma-separated) are allowed - never
  "*". Set it to your teammate's Base44 app domain(s), e.g.
  ALLOWED_ORIGIN=https://your-app.base44.app

Request size
  Bodies larger than MAX_UPLOAD_BYTES (default 12 MB) are rejected with 413
  before the image is decoded or the model runs.

This is a Smart India Hackathon 2026 prototype (PS SIH26038, Team Diasight) -
a screening aid, not a diagnostic device.
"""

from __future__ import annotations

import base64
import binascii
import os

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import DISCLAIMER

# --------------------------------------------------------------------------- #
# config from env
# --------------------------------------------------------------------------- #
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGIN", "").split(",") if o.strip()
]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
DR_ALLOW_UNTRAINED = os.getenv("DR_ALLOW_UNTRAINED", "0") == "1"

app = FastAPI(
    title="DR-Sight ML API",
    version="0.1.0",
    description="Diabetic retinopathy screening aid - not a diagnostic device.",
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
        allow_credentials=False,
    )


# --------------------------------------------------------------------------- #
# middleware: hard body-size cap (before anything reads the body)
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Payload too large (> {MAX_UPLOAD_BYTES} bytes)."},
            )
    return await call_next(request)


# --------------------------------------------------------------------------- #
# auth dependency
# --------------------------------------------------------------------------- #
def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not API_KEY:
        # Fail closed: no key configured means the service is misconfigured.
        raise HTTPException(status_code=503, detail="API_KEY not configured on server.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


# --------------------------------------------------------------------------- #
# lazy pipeline
# --------------------------------------------------------------------------- #
def _pipeline():
    from src.inference.pipeline import get_pipeline

    return get_pipeline(allow_untrained=DR_ALLOW_UNTRAINED)


class PredictJSON(BaseModel):
    image: str  # base64-encoded image bytes (data-URI prefix tolerated)
    with_heatmap: bool = True


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        pipe = _pipeline()
    except Exception as e:  # weights missing and untrained not allowed
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}
    return {
        "status": "ok",
        "model_loaded": True,
        "using_trained_weights": pipe.trained,
        "synthetic_weights": getattr(pipe, "synthetic_weights", False),
        "backbone": pipe.backbone,
        "disclaimer": DISCLAIMER,
    }


def _decode_b64_image(data: str) -> bytes:
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="`image` is not valid base64.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Decoded image too large.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image.")
    return raw


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None),
):
    """Accept either:
      * multipart/form-data with a `file` field (image upload), or
      * application/json  {"image": "<base64>", "with_heatmap": true}
    """
    ctype = request.headers.get("content-type", "")
    with_heatmap = True

    if file is not None:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image too large.")
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")
    elif "application/json" in ctype:
        body = await request.json()
        try:
            payload = PredictJSON(**body)
        except Exception:
            raise HTTPException(status_code=400, detail="Expected JSON with an `image` field.")
        raw = _decode_b64_image(payload.image)
        with_heatmap = payload.with_heatmap
    else:
        raise HTTPException(
            status_code=400,
            detail="Send multipart form-data (`file`) or JSON (`image` base64).",
        )

    try:
        result = _pipeline().run(raw, with_heatmap=with_heatmap)
    except Exception as e:  # decode failure, corrupt image, model error
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    return result.to_dict()
