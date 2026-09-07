"""Non-API smoke tests: quality gate + inference pipeline contract.

Uses an untrained model so no checkpoint is needed. Asserts structure and
referral logic, not accuracy.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("DR_DEVICE", "cpu")
os.environ.setdefault("DR_BACKBONE", "edge")

from src.config import REFERRAL_ACTIONS, PipelineConfig  # noqa: E402
from src.inference.pipeline import DRPipeline  # noqa: E402
from src.quality.image_quality import assess_quality  # noqa: E402


def _fundus(mean_ok=True) -> Image.Image:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    img[(xx - 256) ** 2 + (yy - 256) ** 2 <= 235 ** 2] = (90, 60, 45)
    img = np.clip(img + np.random.randint(0, 20, img.shape, dtype=np.uint8), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def test_quality_accepts_reasonable_fundus():
    r = assess_quality(_fundus())
    assert r["usable"] is True
    assert r["reason"] is None
    assert "blur_var_laplacian" in r["scores"]


@pytest.mark.parametrize(
    "mutate,expect_reason_substr",
    [
        (lambda a: (a * 0.03).astype("uint8"), "xposed"),           # underexposed
        # disc blown to white but dark surround kept -> overexposed, not "cropped"
        (lambda a: np.where(a > 0, 255, 0).astype("uint8"), "xposed"),
        (lambda a: __import__("cv2").GaussianBlur(a, (41, 41), 0), "blurr"),  # blur
    ],
)
def test_quality_rejects_bad_images(mutate, expect_reason_substr):
    arr = np.asarray(_fundus())
    r = assess_quality(mutate(arr))
    assert r["usable"] is False
    assert expect_reason_substr.lower() in r["reason"].lower()


@pytest.fixture(scope="module")
def pipe():
    return DRPipeline(cfg=PipelineConfig(), allow_untrained=True)


def test_pipeline_output_contract(pipe):
    res = pipe.run(_fundus()).to_dict()
    assert res["usable_image"] is True
    assert res["grade"] in range(5)
    assert res["label"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["uncertain"], bool)
    assert isinstance(res["referable"], bool)
    assert 0.0 <= res["p_referable"] <= 1.0
    if res["referral_escalated"]:
        assert res["referable"] is True and res["grade"] < 2
        assert res["referral_action"] == REFERRAL_ACTIONS[2]
    else:
        assert res["referral_action"] == REFERRAL_ACTIONS[res["grade"]]
    assert res["heatmap_base64"]
    assert "not a diagnosis" in res["disclaimer"]


def test_referable_decision_threshold_logic():
    from src.inference.pipeline import referable_decision

    # argmax grade 1, but 0.45 mass on grades >=2
    probs = [0.10, 0.45, 0.30, 0.10, 0.05]
    assert referable_decision(probs, threshold=0.5) == (False, pytest.approx(0.45), False)
    ref, p, esc = referable_decision(probs, threshold=0.35)
    assert ref is True and esc is True and p == pytest.approx(0.45)
    # argmax already referable -> not an escalation
    ref, p, esc = referable_decision([0.1, 0.1, 0.6, 0.1, 0.1], threshold=0.35)
    assert ref is True and esc is False


def test_pipeline_rejects_before_model(pipe):
    flat = Image.fromarray(np.full((512, 512, 3), 120, dtype=np.uint8))
    res = pipe.run(flat).to_dict()
    assert res["usable_image"] is False
    assert res["rejection_reason"]
    assert res["grade"] is None
    assert res["heatmap_base64"] is None


def test_uncertainty_flag_respects_threshold(pipe):
    res = pipe.run(_fundus()).to_dict()
    assert res["uncertain"] == (res["confidence"] < pipe.cfg.uncertainty_threshold)
