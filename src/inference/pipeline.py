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
    referral_action: str | None = None
    heatmap_base64: str | None = None
    probabilities: list[float] | None = None
    quality_scores: dict | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


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
        if with_heatmap:
            cam_res = self.gradcam.explain(
                x[0], target_class=grade, alpha=self.cfg.gradcam_alpha
            )
            heatmap_b64 = cam_res["overlay_b64"]

        # [5] referral + uncertainty
        uncertain = confidence < self.cfg.uncertainty_threshold

        return PredictionResult(
            usable_image=True,
            rejection_reason=None,
            grade=grade,
            label=grade_label(grade),
            confidence=round(confidence, 4),
            uncertain=bool(uncertain),
            referable=bool(grade >= REFERABLE_DR_MIN_GRADE),
            referral_action=referral_action(grade),
            heatmap_base64=heatmap_b64,
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
