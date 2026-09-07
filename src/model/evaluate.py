"""Evaluate a trained checkpoint on the held-out test split.

Reports (see src.model.metrics):
  * HEADLINE: sensitivity + specificity for referable DR (grade >= 2)
  * quadratic weighted kappa
  * per-grade sensitivity, specificity, precision, F1, one-vs-rest AUROC
  * confusion matrix

Also writes:
  * <out>/test_metrics.json
  * <out>/confusion_matrix.png

Usage:
    python -m src.model.evaluate --data data/aptos/labels.csv --weights models/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import BACKBONES, GRADE_LABELS
from src.data.dataset import APTOSDataset
from src.model.architecture import load_checkpoint
from src.model.metrics import compute_all, format_report
from src.model.train import pick_device


@torch.no_grad()
def infer_split(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)
        probs.append(prob.cpu().numpy())
        ps.append(prob.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(probs),
    )


def save_confusion_png(cm, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), [GRADE_LABELS[i] for i in range(5)], rotation=45, ha="right")
    ax.set_yticks(range(5), [GRADE_LABELS[i] for i in range(5)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DR-Sight confusion matrix (test)")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/aptos/labels.csv")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.weights, map_location=str(device))
    model.to(device)
    backbone = ckpt.get("backbone", "b3")
    input_size = BACKBONES[backbone]["input_size"]
    print(f"device={device}  backbone={backbone}  weights={args.weights}")

    ds = APTOSDataset(args.data, args.split, input_size=input_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{args.split} images: {len(ds)}")

    y_true, y_pred, y_prob = infer_split(model, loader, device)
    metrics = compute_all(y_true, y_pred, y_prob)
    print(format_report(metrics))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data).parent / "manifest.json"
    synthetic = False
    if manifest_path.is_file():
        synthetic = json.loads(manifest_path.read_text()).get("synthetic", False)
    metrics["_synthetic_data"] = synthetic
    if synthetic:
        print("\n*** WARNING: evaluated on SYNTHETIC data - these numbers are not real. ***")

    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_png(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    print(f"\nWrote {out_dir/'test_metrics.json'} and {out_dir/'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
