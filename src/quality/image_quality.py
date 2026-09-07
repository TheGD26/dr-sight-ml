"""Image-quality gate for fundus photos.

Runs *before* any image reaches the classifier. A plausible-looking Grad-CAM
on a blurry or badly-lit photo is worse than returning nothing, so we reject
unusable images up front and tell the operator why.

Checks:
  * blur      - variance of the Laplacian (sharp images have high variance)
  * exposure  - mean brightness must sit in a sane band, and not too many
                pixels may be fully clipped to black / white
  * fov       - the bright retinal disc must fill a reasonable fraction of
                the frame (not a tiny off-centre circle, not a full-frame
                crop with no dark surround)

Returns a plain dict:
    {"usable": bool, "reason": str | None, "scores": {...}}
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.config import QualityConfig


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _as_rgb_array(image: "Image.Image | np.ndarray") -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_scores(gray: np.ndarray) -> dict[str, float]:
    mean = float(gray.mean())
    total = gray.size
    dark_frac = float((gray <= 5).sum() / total)
    bright_frac = float((gray >= 250).sum() / total)
    return {
        "mean_brightness": mean,
        "dark_clip_frac": dark_frac,
        "bright_clip_frac": bright_frac,
    }


def fov_fill_fraction(gray: np.ndarray) -> dict[str, float]:
    """Estimate how much of the frame the illuminated retina covers.

    Otsu-threshold the (blurred) grayscale image, take the largest bright
    blob, and compare its area and its min-enclosing-circle radius against
    the frame. Cheap and good enough for a prototype gate.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"fov_fill": 0.0, "fov_radius_ratio": 0.0, "fov_offcenter": 1.0}

    largest = max(contours, key=cv2.contourArea)
    area_frac = float(cv2.contourArea(largest) / (h * w))

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    radius_ratio = float(radius / (min(h, w) / 2.0))  # 1.0 == inscribed circle

    # How far the blob centre sits from the frame centre, normalised.
    offcenter = float(
        np.hypot(cx - w / 2.0, cy - h / 2.0) / (np.hypot(w, h) / 2.0)
    )
    return {
        "fov_fill": area_frac,
        "fov_radius_ratio": radius_ratio,
        "fov_offcenter": offcenter,
    }


def assess_quality(
    image: "Image.Image | np.ndarray | str",
    cfg: QualityConfig | None = None,
) -> dict[str, Any]:
    """Score an image and decide whether it is usable.

    `image` may be a PIL image, an HxWxC RGB array, or a path.
    """
    cfg = cfg or QualityConfig()

    if isinstance(image, str):
        pil = Image.open(image)
        rgb = _as_rgb_array(pil)
    else:
        rgb = _as_rgb_array(image)

    gray = _to_gray_uint8(rgb)

    blur = blur_score(gray)
    exp = exposure_scores(gray)
    fov = fov_fill_fraction(gray)

    scores = {"blur_var_laplacian": blur, **exp, **fov}

    # --- decision (first failing check wins, ordered by how fatal it is) ---
    reason: str | None = None

    if fov["fov_fill"] < cfg.fov_min_fill:
        reason = (
            f"Retina fills only {fov['fov_fill']*100:.0f}% of the frame "
            f"(need >= {cfg.fov_min_fill*100:.0f}%). Move closer / re-centre."
        )
    elif fov["fov_fill"] > cfg.fov_max_fill:
        reason = (
            "No dark surround visible - image looks cropped or is not a "
            "full fundus photo."
        )
    elif exp["mean_brightness"] < cfg.exposure_min_mean:
        reason = (
            f"Underexposed (mean brightness {exp['mean_brightness']:.0f}). "
            "Increase illumination."
        )
    elif exp["mean_brightness"] > cfg.exposure_max_mean:
        reason = (
            f"Overexposed (mean brightness {exp['mean_brightness']:.0f}). "
            "Reduce flash / illumination."
        )
    elif exp["dark_clip_frac"] > cfg.clip_frac_max:
        reason = (
            f"{exp['dark_clip_frac']*100:.0f}% of pixels are pure black - "
            "under-illuminated or wrong field."
        )
    elif exp["bright_clip_frac"] > cfg.clip_frac_max:
        reason = (
            f"{exp['bright_clip_frac']*100:.0f}% of pixels are blown out - "
            "flash glare or overexposure."
        )
    elif blur < cfg.blur_min_variance:
        reason = (
            f"Image too blurry (sharpness {blur:.0f} < {cfg.blur_min_variance:.0f}). "
            "Hold steady / refocus."
        )

    return {"usable": reason is None, "reason": reason, "scores": scores}
