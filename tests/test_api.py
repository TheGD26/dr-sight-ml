"""API smoke tests. Run: PYTHONPATH=. pytest -q

These use an untrained model (DR_ALLOW_UNTRAINED=1) so they don't need a
checkpoint - they assert the *response contract*, not model accuracy.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("API_KEY", "test-key-123")
os.environ.setdefault("DR_ALLOW_UNTRAINED", "1")
os.environ.setdefault("DR_DEVICE", "cpu")
os.environ.setdefault("DR_BACKBONE", "edge")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)
KEY = os.environ["API_KEY"]


def _fake_fundus_png() -> bytes:
    """A dark frame with a bright centred disc - passes the quality gate."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    disc = (xx - 256) ** 2 + (yy - 256) ** 2 <= 235 ** 2
    img[disc] = (90, 60, 45)
    img = np.clip(img + np.random.randint(0, 20, img.shape, dtype=np.uint8), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


EXPECTED_KEYS = {
    "usable_image",
    "rejection_reason",
    "grade",
    "label",
    "confidence",
    "uncertain",
    "referral_action",
    "heatmap_base64",
    "disclaimer",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}


def test_predict_requires_api_key():
    r = client.post("/predict", files={"file": ("f.png", _fake_fundus_png(), "image/png")})
    assert r.status_code == 401


def test_predict_multipart_ok():
    r = client.post(
        "/predict",
        headers={"X-API-Key": KEY},
        files={"file": ("fundus.png", _fake_fundus_png(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert EXPECTED_KEYS.issubset(body.keys())
    assert body["usable_image"] is True
    assert body["grade"] in {0, 1, 2, 3, 4}
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["referral_action"], str) and body["referral_action"]
    assert body["heatmap_base64"] and base64.b64decode(body["heatmap_base64"])
    assert "not a diagnosis" in body["disclaimer"]


def test_predict_json_base64_ok():
    b64 = base64.b64encode(_fake_fundus_png()).decode()
    r = client.post(
        "/predict",
        headers={"X-API-Key": KEY},
        json={"image": b64, "with_heatmap": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert EXPECTED_KEYS.issubset(body.keys())
    assert body["heatmap_base64"] is None  # with_heatmap=False


def test_predict_rejects_blurry_image():
    # near-uniform gray -> fails the blur / FOV gate
    img = Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post(
        "/predict",
        headers={"X-API-Key": KEY},
        files={"file": ("flat.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["usable_image"] is False
    assert body["rejection_reason"]
    assert body["grade"] is None


def test_predict_rejects_garbage_bytes():
    r = client.post(
        "/predict",
        headers={"X-API-Key": KEY},
        files={"file": ("x.png", b"not an image", "image/png")},
    )
    assert r.status_code == 422


def test_predict_oversize_body_413():
    big = b"\x00" * (int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024))) + 1)
    r = client.post(
        "/predict",
        headers={"X-API-Key": KEY, "Content-Type": "application/octet-stream"},
        content=big,
    )
    assert r.status_code == 413


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
