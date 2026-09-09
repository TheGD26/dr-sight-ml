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
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import DISCLAIMER

_LOG = logging.getLogger("dr_sight.api")

# --------------------------------------------------------------------------- #
# config from env
# --------------------------------------------------------------------------- #
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGIN", "").split(",") if o.strip()
]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
DR_ALLOW_UNTRAINED = os.getenv("DR_ALLOW_UNTRAINED", "0") == "1"

# Which screening backend serves /predict.
#   "matlab" (default) - the entire screening computation (image-quality
#            assessment, DR grading, Grad-CAM explainability) runs inside MATLAB
#            via matlab/screen_image.m, called through the MATLAB Engine API for
#            Python. PS SIH26038 requires the MATLAB-based pipeline, so this is
#            the default.
#   "python" - the original PyTorch pipeline (src/inference/pipeline.py). Kept
#            only for local debugging and for comparing against the validated
#            PyTorch numbers (see scripts/compare_backends.py). Not for
#            production under PS26038.
SCREENING_BACKEND = os.getenv("SCREENING_BACKEND", "matlab").strip().lower()
if SCREENING_BACKEND not in ("matlab", "python"):
    raise RuntimeError(
        f"SCREENING_BACKEND must be 'matlab' or 'python', got {SCREENING_BACKEND!r}"
    )

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the active backend on boot so the first /predict doesn't pay the cost.

    For SCREENING_BACKEND=matlab this eagerly starts the MATLAB engine and
    imports the ONNX network ONCE (several seconds); a failure is logged, not
    fatal, so /health can still report the degraded state. The python backend
    loads lazily on first use, as before.
    """
    if SCREENING_BACKEND == "matlab":
        _LOG.info("SCREENING_BACKEND=matlab - starting MATLAB engine + importing ONNX ...")
        t0 = time.perf_counter()
        try:
            from src.inference.matlab_pipeline import get_matlab_pipeline

            h = get_matlab_pipeline().health()
            _LOG.info(
                "MATLAB backend ready in %.1fs (engine %.1fs, ONNX import %.1fs, net_ready=%s)",
                time.perf_counter() - t0,
                h.get("engine_start_seconds") or -1,
                h.get("net_import_seconds") or -1,
                h.get("network_ready"),
            )
        except Exception as e:  # MatlabEngineUnavailable etc. - don't crash boot
            _LOG.error("MATLAB backend failed to start: %s", e)
    else:
        _LOG.info("SCREENING_BACKEND=python - PyTorch pipeline will load lazily.")
    yield


app = FastAPI(
    title="DR-Sight ML API",
    version="0.1.0",
    description="Diabetic retinopathy screening aid - not a diagnostic device.",
    lifespan=lifespan,
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
# lazy pipelines
# --------------------------------------------------------------------------- #
def _python_pipeline():
    """The original PyTorch pipeline. Only used when SCREENING_BACKEND=python."""
    from src.inference.pipeline import get_pipeline

    return get_pipeline(allow_untrained=DR_ALLOW_UNTRAINED)


def _matlab_pipeline():
    """The MATLAB screening backend (default). Owns one MATLAB engine session."""
    from src.inference.matlab_pipeline import get_matlab_pipeline

    return get_matlab_pipeline()


def _screen(raw: bytes, *, with_heatmap: bool) -> dict:
    """Run one image through the active backend and return a PredictionResult-shaped
    dict. Identical output contract for both backends."""
    if SCREENING_BACKEND == "matlab":
        return _matlab_pipeline().run(raw, with_heatmap=with_heatmap)
    return _python_pipeline().run(raw, with_heatmap=with_heatmap).to_dict()


class PredictJSON(BaseModel):
    image: str  # base64-encoded image bytes (data-URI prefix tolerated)
    with_heatmap: bool = True


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    if SCREENING_BACKEND == "matlab":
        try:
            pipe = _matlab_pipeline()
        except Exception as e:  # MATLAB missing / license / ONNX import failure
            return {
                "status": "degraded",
                "screening_backend": "matlab",
                "model_loaded": False,
                "matlab_engine_started": False,
                "onnx_network_imported": False,
                "detail": str(e),
                "disclaimer": DISCLAIMER,
            }
        h = pipe.health()
        return {
            "status": "ok" if h["network_ready"] else "degraded",
            "screening_backend": "matlab",
            "model_loaded": h["network_ready"],
            "matlab_engine_started": h["engine_started"],
            "matlab_engine_start_seconds": h["engine_start_seconds"],
            "onnx_network_imported": h["network_ready"],
            "onnx_import_seconds": h["net_import_seconds"],
            "onnx_path": h["onnx_path"],
            # `using_trained_weights` only means something for the python backend.
            "disclaimer": DISCLAIMER,
        }

    # --- python backend -------------------------------------------------- #
    try:
        pipe = _python_pipeline()
    except Exception as e:  # weights missing and untrained not allowed
        return {
            "status": "degraded",
            "screening_backend": "python",
            "model_loaded": False,
            "detail": str(e),
        }
    return {
        "status": "ok",
        "screening_backend": "python",
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
        result = _screen(raw, with_heatmap=with_heatmap)
    except Exception as e:  # decode failure, corrupt image, model / MATLAB error
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    return result
