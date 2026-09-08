"""Export the trained DR-Sight checkpoint (models/best.pt) to ONNX.

The MATLAB side (SIH PS26038) consumes the ONNX file via
``importNetworkFromONNX``. This script is the single source of truth for the
graph it imports and for the preprocessing constants MATLAB must reproduce.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights models/best.pt --out models/dr_sight.onnx

What it does:
  * loads models/best.pt with src.model.architecture.load_checkpoint
    (MobileNetV3-Small "edge" backbone, 5-class logits head)
  * exports to models/dr_sight.onnx with torch.onnx.export
      - input  name "input",  shape [batch, 3, H, W]  (dynamic batch axis)
      - output name "logits",  shape [batch, 5]        (raw logits, pre-softmax)
  * prints the exact input shape + ImageNet mean/std the MATLAB preprocessing
    must match, and (if onnxruntime is installed) checks ONNX vs PyTorch
    logits agree on a random input.

Nothing else in the repo is touched; the only new artefact is the .onnx file.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# The legacy TorchScript exporter is intentional here (see below); silence its
# migration notice so the printed preprocessing block is the only output.
warnings.filterwarnings("ignore", message=r".*legacy TorchScript-based ONNX export.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.config import BACKBONES, GRADE_LABELS, IMAGENET_MEAN, IMAGENET_STD
from src.model.architecture import load_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"),
                    help="checkpoint saved by src.model.train (default: models/best.pt)")
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "dr_sight.onnx"),
                    help="output .onnx path (default: models/dr_sight.onnx)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"No checkpoint at {weights}. Train first: python -m src.model.train ...")

    model, ckpt = load_checkpoint(str(weights), map_location="cpu")
    model.eval()

    backbone = ckpt.get("backbone", "b3")
    size = int(BACKBONES[backbone]["input_size"])
    num_classes = model.net.get_classifier().out_features

    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)
    with torch.no_grad():
        ref_logits = model(dummy).cpu().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False -> the stable TorchScript exporter. It gives a clean static
    # graph for MobileNetV3 and only needs the `onnx` package (not onnxscript).
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # ---- optional round-trip check -------------------------------------- #
    ort_status = "onnxruntime not installed - skipped ONNX vs PyTorch check"
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        (onnx_logits,) = sess.run(["logits"], {"input": dummy.numpy()})
        max_abs = float(np.abs(onnx_logits - ref_logits).max())

        # dynamic batch axis: run a batch of 4 too
        batch = torch.randn(4, 3, size, size, dtype=torch.float32)
        with torch.no_grad():
            ref_b = model(batch).cpu().numpy()
        (onnx_b,) = sess.run(["logits"], {"input": batch.numpy()})
        max_abs_b = float(np.abs(onnx_b - ref_b).max())
        ort_status = (
            f"ONNX vs PyTorch max|Δlogit|: {max_abs:.2e} (batch=1), "
            f"{max_abs_b:.2e} (batch=4)  -> {'OK' if max(max_abs, max_abs_b) < 1e-3 else 'MISMATCH'}"
        )
    except ImportError:
        pass

    # ---- report ------------------------------------------------------- #
    print("\n" + "=" * 68)
    print(f"  exported: {out_path}")
    print(f"  backbone: {backbone}  ({BACKBONES[backbone]['timm_name']})")
    print("=" * 68)
    print("  MATLAB preprocessing must match EXACTLY:")
    print(f"    input name .......... 'input'")
    print(f"    output name ......... 'logits'   (raw logits - apply softmax in MATLAB)")
    print(f"    input layout ........ NCHW  ->  [batch, 3, {size}, {size}]")
    print(f"    dynamic axis ........ batch (axis 0)")
    print(f"    dtype ............... float32")
    print(f"    channel order ....... RGB")
    print(f"    pixel scale ......... divide uint8 [0,255] by 255.0 first")
    print(f"    ImageNet mean (RGB) . {tuple(round(float(m), 4) for m in IMAGENET_MEAN)}")
    print(f"    ImageNet std  (RGB) . {tuple(round(float(s), 4) for s in IMAGENET_STD)}")
    print(f"    normalize ........... (pixel/255 - mean) / std   per channel")
    print(f"    resize .............. longest side -> {size}, then centre-pad to "
          f"{size}x{size} with 0 (letterbox)")
    print(f"    classes ............. {num_classes}  {list(GRADE_LABELS.values())}")
    print("=" * 68)
    print(f"  {ort_status}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
