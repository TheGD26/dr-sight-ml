"""Unit tests for the MATLAB screening backend bridge.

These exercise the pure-Python contract-conversion layer of
``src/inference/matlab_pipeline.py`` (the part that turns a
``jsonencode(screen_image(...))`` payload into the exact
``PredictionResult.to_dict()`` shape) plus the graceful-degradation behaviour
of ``GET /health`` when ``SCREENING_BACKEND=matlab`` but the MATLAB Engine API
is not installed. None of this needs a local MATLAB.

Run: PYTHONPATH=. pytest -q tests/test_matlab_bridge.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.inference.matlab_pipeline import (
    _RESULT_KEYS,
    MatlabEngineUnavailable,
    _is_empty,
    _normalize_result,
    get_matlab_pipeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The canonical field order of PredictionResult.to_dict() in
# src/inference/pipeline.py. Duplicated here on purpose so a reordering of the
# dataclass is caught by a failing test rather than silently followed.
_PREDICTION_RESULT_KEYS = (
    "usable_image",
    "rejection_reason",
    "grade",
    "label",
    "confidence",
    "uncertain",
    "referable",
    "p_referable",
    "referral_escalated",
    "referral_action",
    "heatmap_base64",
    "affected_regions",
    "probabilities",
    "quality_scores",
    "disclaimer",
)


def _matlab_usable_payload() -> dict:
    """What jsonencode(screen_image(goodImage, net)) looks like for a gradeable
    image (single Grad-CAM region -> MATLAB emits a bare object, not an array)."""
    return json.loads(
        """
        {
          "usable_image": true,
          "rejection_reason": "",
          "grade": 2,
          "label": "Moderate NPDR",
          "confidence": 0.7123,
          "uncertain": false,
          "referable": true,
          "p_referable": 0.8012,
          "referral_escalated": false,
          "referral_action": "Refer to ophthalmologist within ~6 months.",
          "heatmap_base64": "aGVsbG8=",
          "affected_regions": {
            "label": "Model focus region 1",
            "x": 50.0, "y": 48.0, "radius": 12.0, "intensity": 0.9
          },
          "probabilities": [0.01, 0.02, 0.7, 0.2, 0.07],
          "quality_scores": {
            "blur_var": 123.4, "mean_brightness": 90.1,
            "dark_clip_frac": 0.3, "bright_clip_frac": 0.0, "fov_fraction": 0.6
          },
          "disclaimer": "AI screening aid only - not a diagnosis. Confirm with an ophthalmologist."
        }
        """
    )


def _matlab_rejected_payload() -> dict:
    """jsonencode(screen_image(blurryImage, net)) - every gradeable field is a
    MATLAB empty ([] -> [], "" -> "")."""
    return json.loads(
        """
        {
          "usable_image": false,
          "rejection_reason": "Image too blurry (sharpness 12 < 80). Hold steady / refocus.",
          "grade": [],
          "label": "",
          "confidence": [],
          "uncertain": false,
          "referable": [],
          "p_referable": [],
          "referral_escalated": false,
          "referral_action": "",
          "heatmap_base64": "",
          "affected_regions": [],
          "probabilities": [],
          "quality_scores": {
            "blur_var": 12.0, "mean_brightness": 90.0,
            "dark_clip_frac": 0.1, "bright_clip_frac": 0.0, "fov_fraction": 0.6
          },
          "disclaimer": "AI screening aid only - not a diagnosis. Confirm with an ophthalmologist."
        }
        """
    )


# --------------------------------------------------------------------------- #
# _normalize_result: the JSON -> PredictionResult contract
# --------------------------------------------------------------------------- #
def test_result_key_order_matches_prediction_result():
    assert _RESULT_KEYS == _PREDICTION_RESULT_KEYS


def test_normalize_usable_result_shape_and_types():
    out = _normalize_result(_matlab_usable_payload(), with_heatmap=True)

    # exact key set and order of PredictionResult.to_dict()
    assert tuple(out.keys()) == _PREDICTION_RESULT_KEYS

    assert out["usable_image"] is True
    assert out["rejection_reason"] is None  # "" -> None
    assert out["grade"] == 2 and isinstance(out["grade"], int)
    assert out["label"] == "Moderate NPDR"
    assert isinstance(out["confidence"], float) and out["confidence"] == pytest.approx(0.7123)
    assert out["uncertain"] is False
    assert out["referable"] is True
    assert out["p_referable"] == pytest.approx(0.8012)
    assert out["referral_escalated"] is False
    assert out["heatmap_base64"] == "aGVsbG8="
    assert out["probabilities"] == pytest.approx([0.01, 0.02, 0.7, 0.2, 0.07])
    assert all(isinstance(p, float) for p in out["probabilities"])
    assert "not a diagnosis" in out["disclaimer"]


def test_normalize_wraps_single_region_into_a_list():
    out = _normalize_result(_matlab_usable_payload(), with_heatmap=True)
    regions = out["affected_regions"]
    assert isinstance(regions, list) and len(regions) == 1
    r = regions[0]
    assert set(r) == {"label", "x", "y", "radius", "intensity"}
    assert r["label"] == "Model focus region 1"
    assert (r["x"], r["y"], r["radius"], r["intensity"]) == (50.0, 48.0, 12.0, 0.9)
    assert all(isinstance(r[k], float) for k in ("x", "y", "radius", "intensity"))


def test_normalize_rejected_result_maps_empties_to_none():
    out = _normalize_result(_matlab_rejected_payload(), with_heatmap=True)

    assert tuple(out.keys()) == _PREDICTION_RESULT_KEYS
    assert out["usable_image"] is False
    assert out["rejection_reason"].startswith("Image too blurry")
    for k in (
        "grade",
        "label",
        "confidence",
        "referable",
        "p_referable",
        "referral_action",
        "heatmap_base64",
        "affected_regions",
        "probabilities",
    ):
        assert out[k] is None, f"{k} should be None on a rejected image"
    # quality_scores still populated + all floats
    assert out["quality_scores"]["blur_var"] == pytest.approx(12.0)
    assert all(isinstance(v, float) for v in out["quality_scores"].values())


def test_normalize_with_heatmap_false_blanks_cam_fields():
    out = _normalize_result(_matlab_usable_payload(), with_heatmap=False)
    assert out["heatmap_base64"] is None
    assert out["affected_regions"] is None
    # everything else still there
    assert out["grade"] == 2
    assert out["probabilities"] is not None


def test_normalize_handles_scalar_probability_list():
    payload = _matlab_usable_payload()
    payload["probabilities"] = 0.7  # MATLAB 1x1 -> bare number
    out = _normalize_result(payload, with_heatmap=False)
    assert out["probabilities"] == [0.7]


@pytest.mark.parametrize("value", [None, [], "", {}, [[]]])
def test_is_empty_true(value):
    assert _is_empty(value)


@pytest.mark.parametrize("value", [0, 0.0, False, [0.0], "0", {"a": 1}])
def test_is_empty_false(value):
    assert not _is_empty(value)


# --------------------------------------------------------------------------- #
# graceful degradation when the MATLAB Engine API is not installed
# --------------------------------------------------------------------------- #
_HAS_MATLAB_ENGINE = False
try:  # pragma: no cover - depends on the host
    import matlab.engine  # noqa: F401

    _HAS_MATLAB_ENGINE = True
except Exception:
    pass


@pytest.mark.skipif(_HAS_MATLAB_ENGINE, reason="matlab.engine is installed on this host")
def test_get_matlab_pipeline_raises_clear_error_without_engine():
    with pytest.raises(MatlabEngineUnavailable) as ei:
        get_matlab_pipeline()
    msg = str(ei.value)
    assert "matlab.engine" in msg
    assert "extern/engines/python" in msg


@pytest.mark.skipif(_HAS_MATLAB_ENGINE, reason="matlab.engine is installed on this host")
def test_health_matlab_backend_reports_degraded_without_engine():
    """A fresh process with SCREENING_BACKEND=matlab must still boot and answer
    /health with status=degraded (never a 500) when MATLAB is absent."""
    script = textwrap.dedent(
        """
        from fastapi.testclient import TestClient
        from src.api.main import app
        with TestClient(app) as c:          # `with` -> lifespan warmup runs
            r = c.get("/health")
        print(r.status_code)
        print(r.text)
        """
    )
    env = {**os.environ, "SCREENING_BACKEND": "matlab", "PYTHONPATH": str(_REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    status_line, body_line = proc.stdout.strip().splitlines()[:2]
    assert status_line == "200"
    body = json.loads(body_line)
    assert body["status"] == "degraded"
    assert body["screening_backend"] == "matlab"
    assert body["model_loaded"] is False
    assert body["matlab_engine_started"] is False
    assert "matlab.engine" in body["detail"]


# --------------------------------------------------------------------------- #
# end-to-end through a real MATLAB engine (opt-in: slow, needs the ONNX
# converter support package). Enable with DR_RUN_MATLAB_E2E=1.
# --------------------------------------------------------------------------- #
_E2E = os.getenv("DR_RUN_MATLAB_E2E") == "1"
_ONNX = _REPO_ROOT / "models" / "dr_sight.onnx"
_GRADEABLE = _REPO_ROOT / "data" / "aptos" / "cache" / "00e4ddff966a.png"


@pytest.mark.skipif(
    not (_E2E and _HAS_MATLAB_ENGINE and _ONNX.is_file() and _GRADEABLE.is_file()),
    reason="set DR_RUN_MATLAB_E2E=1 with matlab.engine + models/dr_sight.onnx to run",
)
def test_screen_image_end_to_end_via_bridge():
    """Drive matlab/screen_image.m through the bridge on one gradeable fundus
    image and assert the result is a well-formed PredictionResult (regression
    guard for the pngBase64 temp-file bug)."""
    pipe = get_matlab_pipeline()
    raw = _GRADEABLE.read_bytes()

    r = pipe.run(raw, with_heatmap=True)
    assert tuple(r.keys()) == _PREDICTION_RESULT_KEYS
    assert r["usable_image"] is True
    assert r["grade"] in (0, 1, 2, 3, 4)
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["probabilities"]) == 5
    assert abs(sum(r["probabilities"]) - 1.0) < 1e-3
    assert isinstance(r["heatmap_base64"], str) and len(r["heatmap_base64"]) > 1000
    import base64 as _b64

    _b64.b64decode(r["heatmap_base64"])  # must be valid base64
    assert isinstance(r["affected_regions"], list) and r["affected_regions"]
    for reg in r["affected_regions"]:
        assert set(reg) == {"label", "x", "y", "radius", "intensity"}

    # with_heatmap=False must blank the CAM fields, like DRPipeline.run(...)
    r2 = pipe.run(raw, with_heatmap=False)
    assert r2["heatmap_base64"] is None and r2["affected_regions"] is None
    assert r2["grade"] == r["grade"]


@pytest.mark.skipif(
    not (_E2E and _HAS_MATLAB_ENGINE and _ONNX.is_file()),
    reason="set DR_RUN_MATLAB_E2E=1 with matlab.engine + models/dr_sight.onnx to run",
)
def test_matlab_backend_matches_pytorch_numbers():
    """The MATLAB grade / confidence / p_referable must track the reference
    PyTorch pipeline within tolerance on gradeable images."""
    from src.config import PipelineConfig
    from src.inference.pipeline import DRPipeline

    py = DRPipeline(cfg=PipelineConfig())
    if not py.trained:
        pytest.skip("no trained PyTorch checkpoint to compare against")
    ml = get_matlab_pipeline()

    cache = sorted((_REPO_ROOT / "data" / "aptos" / "cache").glob("*.png"))
    graded = 0
    for img in cache:
        if graded >= 5:
            break
        raw = img.read_bytes()
        rp = py.run(raw, with_heatmap=False).to_dict()
        if not rp["usable_image"]:
            continue
        rm = ml.run(raw, with_heatmap=False)
        assert rm["usable_image"] is True
        assert rm["grade"] == rp["grade"], f"{img.name}: {rm['grade']} vs {rp['grade']}"
        max_dp = max(abs(a - b) for a, b in zip(rp["probabilities"], rm["probabilities"]))
        assert max_dp < 0.02, f"{img.name}: max |Δ softmax| {max_dp:.4f}"
        assert abs(rm["confidence"] - rp["confidence"]) < 0.02
        assert abs(rm["p_referable"] - rp["p_referable"]) < 0.02
        graded += 1
    assert graded > 0, "no gradeable images found in the cache"
