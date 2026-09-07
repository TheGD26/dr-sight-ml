"""Transfer-learning trainer for DR-Sight.

Two-stage schedule:
  Stage 1  freeze backbone, train the 5-way head        (lr_head, epochs_head)
  Stage 2  unfreeze everything, fine-tune               (lr_finetune, epochs_finetune)

Loss:  class-weighted cross-entropy (APTOS grade 0 dominates).
Ckpt:  best validation quadratic weighted kappa is saved to <out_dir>/best.pt.
Log:   train/val loss + per-class metrics every epoch, to stdout and
       <out_dir>/train_log.jsonl.

Usage:
    python -m src.model.train --data data/aptos/labels.csv --backbone b3
    python -m src.model.train --data data/aptos/labels.csv --backbone edge \
        --epochs-head 1 --epochs-finetune 1 --batch-size 8   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import BACKBONES, TrainConfig
from src.data.dataset import APTOSDataset
from src.model.architecture import build_model
from src.model.metrics import compute_all, format_report


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    losses, ys, ps, probs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        prob = torch.softmax(logits, dim=1)
        probs.append(prob.cpu().numpy())
        ps.append(prob.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs)
    m = compute_all(y_true, y_pred, y_prob)
    m["loss"] = float(np.mean(losses))
    return m


def run_stage(
    model, name, loaders, device, criterion, epochs, lr, weight_decay, log_fp, best,
    extra_ckpt: dict | None = None,
):
    extra_ckpt = extra_ckpt or {}
    train_loader, val_loader = loaders
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running.append(loss.item())
        sched.step()

        val = evaluate(model, val_loader, device, criterion)
        train_loss = float(np.mean(running))
        rec = {
            "stage": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_qwk": val["quadratic_weighted_kappa"],
            "val_referable_sensitivity": val["referable_dr"]["sensitivity"],
            "val_accuracy": val["accuracy"],
            "per_grade_sensitivity": {g: r["sensitivity"] for g, r in val["per_grade"].items()},
            "secs": round(time.time() - t0, 1),
        }
        log_fp.write(json.dumps(rec) + "\n")
        log_fp.flush()
        print(
            f"[{name} {epoch}/{epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} val_QWK={rec['val_qwk']:.4f} "
            f"referable_sens={rec['val_referable_sensitivity']*100:.1f}% "
            f"({rec['secs']}s)"
        )

        if rec["val_qwk"] > best["val_qwk"]:
            best.update(val_qwk=rec["val_qwk"], stage=name, epoch=epoch, metrics=val)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": model.backbone_key,
                    "metrics": val,
                    "stage": name,
                    "epoch": epoch,
                    **extra_ckpt,
                },
                best["path"],
            )
            print(f"    -> new best QWK {rec['val_qwk']:.4f}; saved {best['path']}")
    return best


def main() -> None:
    tc = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=f"{tc.data_dir}/labels.csv")
    ap.add_argument("--backbone", default=tc.backbone, choices=list(BACKBONES))
    ap.add_argument("--out", default=tc.out_dir)
    ap.add_argument("--epochs-head", type=int, default=tc.epochs_head)
    ap.add_argument("--epochs-finetune", type=int, default=tc.epochs_finetune)
    ap.add_argument("--batch-size", type=int, default=tc.batch_size)
    ap.add_argument("--lr-head", type=float, default=tc.lr_head)
    ap.add_argument("--lr-finetune", type=float, default=tc.lr_finetune)
    ap.add_argument("--num-workers", type=int, default=tc.num_workers)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=tc.seed)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: cap train batches")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    input_size = BACKBONES[args.backbone]["input_size"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.data).parent / "manifest.json"
    synthetic_data = False
    if manifest_path.is_file():
        synthetic_data = json.loads(manifest_path.read_text()).get("synthetic", False)
    if synthetic_data:
        print("!!! Training on SYNTHETIC placeholder data - the checkpoint is for")
        print("!!! pipeline testing only; its metrics are not real.")
    print(f"device={device}  backbone={args.backbone}  input_size={input_size}")

    train_ds = APTOSDataset(args.data, "train", input_size=input_size)
    val_ds = APTOSDataset(args.data, "val", input_size=input_size)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    if args.limit_train:
        from itertools import islice

        class _Cap:
            def __init__(self, dl, n): self.dl, self.n = dl, n
            def __iter__(self): return islice(iter(self.dl), self.n)
            def __len__(self): return min(self.n, len(self.dl))

        train_loader = _Cap(train_loader, args.limit_train)

    model = build_model(backbone=args.backbone, pretrained=True).to(device)

    if args.no_class_weights:
        weight = None
    else:
        weight = train_ds.class_weights().to(device)
        print(f"class weights: {weight.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    best = {"val_qwk": -1.0, "path": str(out_dir / "best.pt"), "stage": None, "epoch": 0}
    extra_ckpt = {"synthetic_data": synthetic_data, "train_args": vars(args)}
    log_path = out_dir / "train_log.jsonl"
    with open(log_path, "w") as log_fp:
        # Stage 1: head only
        model.freeze_backbone()
        best = run_stage(
            model, "head", (train_loader, val_loader), device, criterion,
            args.epochs_head, args.lr_head, tc.weight_decay, log_fp, best, extra_ckpt,
        )
        # Stage 2: full fine-tune
        model.unfreeze_all()
        best = run_stage(
            model, "finetune", (train_loader, val_loader), device, criterion,
            args.epochs_finetune, args.lr_finetune, tc.weight_decay, log_fp, best, extra_ckpt,
        )

    if best["val_qwk"] < 0:
        print("WARNING: no checkpoint met the bar; saving last-epoch weights as fallback.")
        torch.save(
            {"model_state": model.state_dict(), "backbone": model.backbone_key,
             "metrics": {}, "stage": "last", "epoch": -1, **extra_ckpt},
            best["path"],
        )
    else:
        print(f"\nBest: {best['stage']} epoch {best['epoch']}  val_QWK={best['val_qwk']:.4f}")
        print(format_report(best["metrics"]))
    print(f"Checkpoint: {best['path']}")
    print(f"Log:        {log_path}")


if __name__ == "__main__":
    main()
