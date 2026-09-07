"""Model definitions for DR-Sight.

Two configs, selected by name (see src.config.BACKBONES):

  * "b3"   - EfficientNet-B3, ImageNet-pretrained. The accuracy default from
             the SIH tech-stack slide. ~12M params; CPU inference is a few
             seconds per image, fine for a demo.

  * "edge" - MobileNetV3-Small, ImageNet-pretrained. ~2.5M params. This is the
             variant that would eventually be quantised to INT8 / TFLite for
             the offline on-device path in the SIH deck. That conversion step
             (torch -> ONNX -> TFLite, post-training INT8 quant, on-device
             Grad-CAM) is OUT OF SCOPE for this prototype - we only make sure
             the small model trains and serves through the same API.

Both use a plain 5-way linear head (ordinal-regression heads are a possible
later improvement; quadratic weighted kappa already rewards getting close).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.config import BACKBONES, NUM_CLASSES


class DRModel(nn.Module):
    def __init__(
        self,
        backbone: str = "b3",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {list(BACKBONES)}")
        self.backbone_key = backbone
        timm_name = BACKBONES[backbone]["timm_name"]

        self.net = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    # --- training helpers ---------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.net.parameters():
            p.requires_grad_(False)
        for p in self.classifier_parameters():
            p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for p in self.net.parameters():
            p.requires_grad_(True)

    def classifier_parameters(self):
        head = self.net.get_classifier()
        return head.parameters()

    # --- introspection for Grad-CAM --------------------------------------- #
    def gradcam_target_layer(self) -> nn.Module:
        """Last conv block of the backbone - what pytorch-grad-cam should hook.

        We target `backbone.blocks[-1]` (the final inverted-residual / MBConv
        stage) rather than the `conv_head` 1x1 projection: for MobileNetV3 the
        post-`conv_head` Hardswish makes the CAM collapse to all-zeros, while
        `blocks[-1]` gives a clean map for both EfficientNet and MobileNetV3.
        Fall back to the last nn.Conv2d found by traversal.
        """
        blocks = getattr(self.net, "blocks", None)
        if isinstance(blocks, nn.Module) and len(blocks) > 0:
            return blocks[-1]
        mod = getattr(self.net, "conv_head", None)
        if isinstance(mod, nn.Module):
            return mod
        last_conv = None
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM target")
        return last_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(backbone: str = "b3", pretrained: bool = True, **kw) -> DRModel:
    return DRModel(backbone=backbone, pretrained=pretrained, **kw)


def load_checkpoint(path: str, map_location: str = "cpu") -> tuple[DRModel, dict]:
    """Load a checkpoint saved by train.py.

    Checkpoint dict: {"model_state", "backbone", "metrics", "config", ...}
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    backbone = ckpt.get("backbone", "b3")
    model = build_model(backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
