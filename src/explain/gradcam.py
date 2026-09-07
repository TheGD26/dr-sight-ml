"""Grad-CAM explainability for DR-Sight.

Wraps `pytorch-grad-cam`, hooking the last convolutional block of the backbone
(EfficientNet `.conv_head` / MobileNetV3 `.conv_head`, resolved by the model).

`explain()` returns:
  * overlay      - PIL.Image, the input blended with the CAM colormap
  * overlay_b64  - base64 PNG string of that overlay (what the API ships)
  * cam          - raw HxW float32 array in [0, 1] (for later lesion-overlap
                   scoring against IDRiD pixel masks)
  * target_class - the class the CAM was computed for

There is also a CLI that runs Grad-CAM on N validation images and dumps the
overlays to tests/gradcam_samples/ so you can eyeball that the heatmap lands
on retinal structures, not the black border or lens artifacts.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import IMAGENET_MEAN, IMAGENET_STD


def _denorm(chw: torch.Tensor) -> np.ndarray:
    """CHW normalised tensor -> HxWx3 float RGB in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (chw.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_cam(rgb01: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] RGB image with a [0,1] CAM using the JET colormap."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - alpha) * rgb01 + alpha * heat
    return np.clip(blended, 0, 1)


def to_b64_png(img_uint8: np.ndarray) -> str:
    pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DRGradCAM:
    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self._target_layers = [model.gradcam_target_layer()]

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        alpha: float = 0.5,
    ) -> dict:
        """input_tensor: normalised CHW or 1CHW tensor."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        if target_class is None:
            with torch.no_grad():
                target_class = int(self.model(input_tensor).argmax(1).item())

        targets = [ClassifierOutputTarget(target_class)]
        # GradCAM needs grad enabled; it manages that internally.
        with GradCAM(model=self.model, target_layers=self._target_layers) as cam_engine:
            grayscale = cam_engine(input_tensor=input_tensor, targets=targets)[0]  # HxW [0,1]

        rgb01 = _denorm(input_tensor[0])
        if grayscale.shape != rgb01.shape[:2]:
            grayscale = cv2.resize(grayscale, (rgb01.shape[1], rgb01.shape[0]))
        blended = overlay_cam(rgb01, grayscale, alpha=alpha)
        overlay_uint8 = np.uint8(255 * blended)

        return {
            "overlay": Image.fromarray(overlay_uint8),
            "overlay_b64": to_b64_png(overlay_uint8),
            "cam": grayscale.astype(np.float32),
            "target_class": target_class,
        }


# --------------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------------- #
def _cli() -> None:
    from torch.utils.data import DataLoader

    from src.config import BACKBONES
    from src.data.dataset import APTOSDataset
    from src.model.architecture import build_model, load_checkpoint

    ap = argparse.ArgumentParser(description="Dump Grad-CAM overlays for eyeballing.")
    ap.add_argument("--data", default="data/aptos/labels.csv")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="tests/gradcam_samples")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.weights).is_file():
        model, ckpt = load_checkpoint(args.weights, map_location=args.device)
        backbone = ckpt.get("backbone", "b3")
        print(f"loaded {args.weights} (backbone={backbone})")
    else:
        backbone = "edge"
        model = build_model(backbone=backbone, pretrained=True)
        print(f"no checkpoint at {args.weights}; using an untrained {backbone} model")

    input_size = BACKBONES[backbone]["input_size"]
    ds = APTOSDataset(args.data, args.split, input_size=input_size, train_aug=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    cam = DRGradCAM(model, device=args.device)
    for i, (x, y) in enumerate(loader):
        if i >= args.n:
            break
        res = cam.explain(x[0])
        fname = out_dir / f"{args.split}_{i:02d}_true{int(y.item())}_pred{res['target_class']}.png"
        res["overlay"].save(fname)
        print(f"  wrote {fname}   cam[min={res['cam'].min():.2f} max={res['cam'].max():.2f}]")

    print(f"\nDone. Eyeball the overlays in {out_dir}/ - the hot region should sit on")
    print("retinal structures, not the black border or lens flare.")


if __name__ == "__main__":
    _cli()
