"""Shared configuration for DR-Sight.

Single source of truth for the DR grading scale, referral policy, model
defaults, and pipeline thresholds. Everything that another module might
otherwise hardcode as a magic string or number lives here.

DR-Sight is a Smart India Hackathon 2026 prototype (PS SIH26038, Team Diasight).
It is NOT a certified medical device. See src/inference/pipeline.py for the
full disclaimer that ships in every API response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# DR grading scale (International Clinical DR severity scale, 5 classes)
# --------------------------------------------------------------------------- #
NUM_CLASSES = 5

# Grade -> human-readable label.
GRADE_LABELS: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

# Grade -> referral action. Sourced from the Diasight project research notes;
# keep this as data, not scattered string literals.
REFERRAL_ACTIONS: dict[int, str] = {
    0: "No referral. Routine annual screening.",
    1: "No urgent referral. Re-screen in 9-12 months.",
    2: "Refer to ophthalmologist within ~6 months.",
    3: "Refer within ~1 month - treat as priority.",
    4: "Urgent referral within days.",
}

# Grades at or above this threshold are "referable DR" — the headline
# sensitivity number for a screening tool.
REFERABLE_DR_MIN_GRADE = 2

DISCLAIMER = (
    "AI screening aid only - not a diagnosis. "
    "Confirm with an ophthalmologist."
)


def grade_label(grade: int) -> str:
    return GRADE_LABELS[int(grade)]


def referral_action(grade: int) -> str:
    return REFERRAL_ACTIONS[int(grade)]


# --------------------------------------------------------------------------- #
# Image / model defaults
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 512          # cached / preprocessed square size
MODEL_INPUT_SIZE = 300    # EfficientNet-B3 native train resolution
CACHE_SIZE = 512          # prepare_aptos.py resize target

# ImageNet normalisation (backbones are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbone configs. "b3" is the accuracy default from the SIH tech-stack slide;
# "edge" is the small variant that would later be quantised to INT8 / TFLite
# for the on-device path (that conversion is out of scope here).
BACKBONES: dict[str, dict] = {
    "b3": {
        "timm_name": "efficientnet_b3",
        "input_size": 300,
    },
    "edge": {
        "timm_name": "mobilenetv3_small_100",
        "input_size": 224,
    },
}
DEFAULT_BACKBONE = "b3"


# --------------------------------------------------------------------------- #
# Pipeline thresholds (env-overridable so deployment can tune without a rebuild)
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    backbone: str = os.getenv("DR_BACKBONE", DEFAULT_BACKBONE)
    weights_path: str = os.getenv("DR_WEIGHTS", "models/best.pt")
    # Top-1 softmax below this -> flag "uncertain": true for mandatory human review.
    uncertainty_threshold: float = float(os.getenv("DR_UNCERTAINTY_THRESHOLD", "0.6"))
    device: str = os.getenv("DR_DEVICE", "auto")  # auto | cpu | cuda | mps
    gradcam_alpha: float = 0.5  # heatmap blend weight

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


# --------------------------------------------------------------------------- #
# Image-quality gate thresholds
# --------------------------------------------------------------------------- #
@dataclass
class QualityConfig:
    # Variance-of-Laplacian below this => too blurry. Tuned for ~512px input.
    blur_min_variance: float = float(os.getenv("DR_BLUR_MIN_VAR", "80.0"))
    # Mean brightness (0-255) must sit inside this band.
    exposure_min_mean: float = float(os.getenv("DR_EXPOSURE_MIN", "25.0"))
    exposure_max_mean: float = float(os.getenv("DR_EXPOSURE_MAX", "230.0"))
    # Fraction of frame the retinal disc must cover.
    fov_min_fill: float = float(os.getenv("DR_FOV_MIN_FILL", "0.20"))
    fov_max_fill: float = float(os.getenv("DR_FOV_MAX_FILL", "0.95"))
    # Clipped-pixel fraction that still counts as over/under exposed even if
    # the mean looks fine.
    clip_frac_max: float = float(os.getenv("DR_CLIP_FRAC_MAX", "0.35"))


# --------------------------------------------------------------------------- #
# Training defaults
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    backbone: str = DEFAULT_BACKBONE
    data_dir: str = "data/aptos"
    out_dir: str = "models"
    epochs_head: int = 3        # frozen-backbone warmup
    epochs_finetune: int = 12   # unfrozen fine-tune
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    use_class_weights: bool = True
    monitor: str = "val_qwk"   # checkpoint selection metric
