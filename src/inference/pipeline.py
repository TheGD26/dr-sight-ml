"""End-to-end inference pipeline for DR-Sight.

    raw image bytes / PIL / path
        |
        v
  [1] image-quality gate   --reject-->  {"usable_image": false, "rejection_reason": ...}
        |
        v
  [2] preprocess (resize, pad, ImageNet-normalise)
        |
        v
  [3] model forward pass -> softmax
        |
        v
  [4] Grad-CAM overlay on the predicted grade
        |
        v
  [5] referral action lookup + uncertainty flag
        |
        v
   JSON result (see PredictionResult.to_dict / the shape in the README)

IMPORTANT - this is a Smart India Hackathon 2026 PROTOTYPE (PS SIH26038, Team
Diasight), NOT a certified or regulated medical device. It SCREENS fundus
images and RECOMMENDS a referral timeframe; it does not diagnose. Every result
carries the disclaimer from src.config.DISCLAIMER and must be confirmed by an
ophthalmologist.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import (
    DISCLAIMER,
    REFERABLE_DR_MIN_GRADE,
    PipelineConfig,
    QualityConfig,
    grade_label,
    referral_action,
)
from src.data.dataset import build_transforms
from src.explain.gradcam import DRGradCAM
from src.model.architecture import build_model, load_checkpoint
from src.quality.image_quality import assess_quality
from src.config import BACKBONES


@dataclass
class PredictionResult:
    usable_image: bool
    rejection_reason: str | None = None
    grade: int | None = None
    label: str | None = None
    confidence: float | None = None
    uncertain: bool = False
    referable: bool | None = None
    p_referable: float | None = None
    referral_escalated: bool = False
    referral_action: str | None = None
    heatmap_base64: str | None = None
    affected_regions: list[dict] | None = None
    probabilities: list[float] | None = None
    quality_scores: dict | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def referable_decision(
    probs, threshold: float, min_grade: int = REFERABLE_DR_MIN_GRADE
) -> tuple[bool, float, bool]:
    """Screening-biased referral decision.

    Returns (referable, p_referable, escalated) where:
      * p_referable = sum of softmax mass on grades >= min_grade
      * referable   = p_referable >= threshold
      * escalated   = referable is driven by the threshold rather than the
                      argmax grade (i.e. argmax grade < min_grade)
    """
    probs = np.asarray(probs, dtype=float)
    grade = int(probs.argmax())
    p_ref = float(probs[min_grade:].sum())
    referable = p_ref >= threshold
    escalated = referable and grade < min_grade
    return referable, p_ref, escalated


def cam_regions(
    cam: np.ndarray, max_regions: int = 4, rel_threshold: float = 0.55
) -> list[dict]:
    """Turn a Grad-CAM heatmap into up to `max_regions` circular regions of
    interest, in the percentage-coordinate shape the Base44 UI expects:

        {"label", "x", "y", "radius", "intensity"}

    x / y are the blob centroid as 0-100 percent of width / height (top-left
    origin); radius is the area-equivalent radius as 0-100 percent of width
    (clamped 2-25); intensity is the mean CAM activation inside the blob (0-1).

    Blobs are the connected components of `cam >= rel_threshold * cam.max()`,
    ranked by (area * mean activation). Grounded entirely in the model's own
    Grad-CAM - no lesion detection is claimed.
    """
    import cv2

    if cam is None or cam.size == 0:
        return []
    cam = np.nan_to_num(cam.astype(np.float32))
    peak = float(cam.max())
    if peak <= 1e-6:
        return []
    h, w = cam.shape
    mask = (cam >= rel_threshold * peak).astype(np.uint8)
    if mask.sum() == 0:
        return []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(9.0, 0.002 * h * w)  # ignore specks
    cand: list[tuple[float, dict]] = []
    for i in range(1, n):  # 0 is background
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[i]
        blob = labels == i
        intensity = float(cam[blob].mean())
        radius_px = float(np.sqrt(area / np.pi))
        region = {
            "label": "",  # filled after ranking
            "x": round(cx / w * 100.0, 1),
            "y": round(cy / h * 100.0, 1),
            "radius": round(min(25.0, max(2.0, radius_px / w * 100.0)), 1),
            "intensity": round(min(1.0, max(0.0, intensity)), 3),
        }
        cand.append((area * intensity, region))

    cand.sort(key=lambda t: t[0], reverse=True)
    out = []
    for rank, (_, region) in enumerate(cand[:max_regions], start=1):
        region["label"] = f"Model focus region {rank}"
        out.append(region)
    return out


def _load_image(src: "bytes | str | Path | Image.Image") -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (str, Path)):
        return Image.open(src).convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    raise TypeError(f"Unsupported image source type: {type(src)}")


class DRPipeline:
    """Loads the model + Grad-CAM once, then `.run()` per image."""

    def __init__(
        self,
        cfg: PipelineConfig | None = None,
        quality_cfg: QualityConfig | None = None,
        allow_untrained: bool = False,
    ):
        self.cfg = cfg or PipelineConfig()
        self.quality_cfg = quality_cfg or QualityConfig()
        self.device = self.cfg.resolved_device()

        weights = Path(self.cfg.weights_path)
        if weights.is_file():
            self.model, ckpt = load_checkpoint(str(weights), map_location=self.device)
            self.backbone = ckpt.get("backbone", self.cfg.backbone)
            self.trained = True
            # True when the checkpoint was trained on the synthetic placeholder
            # dataset - predictions are structurally valid but meaningless.
            self.synthetic_weights = bool(ckpt.get("synthetic_data", False))
        elif allow_untrained:
            self.backbone = self.cfg.backbone
            self.model = build_model(backbone=self.backbone, pretrained=False)
            self.trained = False
            self.synthetic_weights = False
        else:
            raise FileNotFoundError(
                f"No weights at {weights}. Train first (python -m src.model.train ...) "
                "or construct DRPipeline(allow_untrained=True) for smoke tests."
            )

        self.model.to(self.device).eval()
        self.input_size = BACKBONES[self.backbone]["input_size"]
        self.tf = build_transforms(self.input_size, train=False)
        self.gradcam = DRGradCAM(self.model, device=self.device)

    # ------------------------------------------------------------------ #
    def run(self, image_src, with_heatmap: bool = True) -> PredictionResult:
        pil = _load_image(image_src)

        # [1] quality gate
        q = assess_quality(pil, cfg=self.quality_cfg)
        if not q["usable"]:
            return PredictionResult(
                usable_image=False,
                rejection_reason=q["reason"],
                quality_scores=q["scores"],
            )

        # [2] preprocess
        rgb = np.asarray(pil)
        x = self.tf(image=rgb)["image"].unsqueeze(0).to(self.device)

        # [3] forward pass
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        grade = int(probs.argmax())
        confidence = float(probs[grade])

        # [4] Grad-CAM on the predicted grade
        heatmap_b64 = None
        regions: list[dict] | None = None
        if with_heatmap:
            cam_res = self.gradcam.explain(
                x[0], target_class=grade, alpha=self.cfg.gradcam_alpha
            )
            heatmap_b64 = cam_res["overlay_b64"]
            regions = cam_regions(cam_res["cam"])

        # [5] referral + uncertainty
        uncertain = confidence < self.cfg.uncertainty_threshold
        referable, p_referable, escalated = referable_decision(
            probs, self.cfg.referable_threshold
        )
        # `grade` stays the model's argmax point estimate; only the referral
        # action escalates when the threshold (not argmax) triggers referable.
        action_grade = REFERABLE_DR_MIN_GRADE if escalated else grade

        return PredictionResult(
            usable_image=True,
            rejection_reason=None,
            grade=grade,
            label=grade_label(grade),
            confidence=round(confidence, 4),
            uncertain=bool(uncertain),
            referable=bool(referable),
            p_referable=round(p_referable, 4),
            referral_escalated=bool(escalated),
            referral_action=referral_action(action_grade),
            heatmap_base64=heatmap_b64,
            affected_regions=regions,
            probabilities=[round(float(p), 4) for p in probs],
            quality_scores=q["scores"],
        )


# module-level lazy singleton for the API
_PIPELINE: DRPipeline | None = None


def get_pipeline(**kw) -> DRPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = DRPipeline(**kw)
    return _PIPELINE


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the DR-Sight pipeline on one image.")
    ap.add_argument("image")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--no-heatmap", action="store_true")
    ap.add_argument("--allow-untrained", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.weights:
        cfg.weights_path = args.weights
    pipe = DRPipeline(cfg=cfg, allow_untrained=args.allow_untrained)
    res = pipe.run(args.image, with_heatmap=not args.no_heatmap).to_dict()
    if res.get("heatmap_base64"):
        res["heatmap_base64"] = res["heatmap_base64"][:32] + f"... ({len(res['heatmap_base64'])} chars)"
    print(json.dumps(res, indent=2))
