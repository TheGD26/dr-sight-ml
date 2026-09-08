"""Dump per-image PyTorch predictions to a CSV the MATLAB side validates against.

``src.model.evaluate`` only writes aggregate metrics (test_metrics.json); the
MATLAB ONNX round-trip check in matlab/validate_grading.m needs *per-image*
reference predictions. This script produces them with the SAME model and the
SAME preprocessing (src.data.dataset.APTOSDataset) that evaluate.py uses, so
any disagreement the MATLAB script reports is a real ONNX/import problem and
not a preprocessing drift.

    python scripts/make_reference_csv.py
    python scripts/make_reference_csv.py --split test --limit 64 \
        --out matlab/reference_test.csv

CSV columns:
    id_code, abs_path, true_grade,
    pt_grade, pt_label, pt_confidence, pt_p_referable,
    pt_p0, pt_p1, pt_p2, pt_p3, pt_p4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import BACKBONES, GRADE_LABELS, REFERABLE_DR_MIN_GRADE
from src.data.dataset import build_transforms
from src.model.architecture import load_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "aptos" / "labels.csv"))
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all images in the split")
    ap.add_argument("--out", default=str(REPO_ROOT / "matlab" / "reference_test.csv"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model, ckpt = load_checkpoint(args.weights, map_location="cpu")
    model.eval()
    backbone = ckpt.get("backbone", "b3")
    size = int(BACKBONES[backbone]["input_size"])
    tf = build_transforms(size, train=False)

    data_csv = Path(args.data)
    df = pd.read_csv(data_csv)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"No rows for split={args.split!r} in {data_csv}")

    import cv2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    correct = 0
    with torch.no_grad():
        for _, r in df.iterrows():
            p = Path(r["image_path"])
            abs_path = p if p.is_absolute() else (data_csv.parent / p)
            bgr = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(abs_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = tf(image=rgb)["image"].unsqueeze(0)
            probs = torch.softmax(model(x), dim=1)[0].numpy()
            grade = int(probs.argmax())
            p_ref = float(probs[REFERABLE_DR_MIN_GRADE:].sum())
            true_grade = int(r["diagnosis"])
            correct += int(grade == true_grade)
            rows.append(
                {
                    "id_code": r["id_code"],
                    "abs_path": str(abs_path.resolve()),
                    "true_grade": true_grade,
                    "pt_grade": grade,
                    "pt_label": GRADE_LABELS[grade],
                    "pt_confidence": round(float(probs[grade]), 6),
                    "pt_p_referable": round(p_ref, 6),
                    **{f"pt_p{i}": round(float(probs[i]), 6) for i in range(len(probs))},
                }
            )

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_path}  ({len(rows)} rows, split={args.split})")
    print(f"PyTorch accuracy on these rows: {correct}/{len(rows)} = {correct/len(rows):.3f}")


if __name__ == "__main__":
    main()
