"""Prepare the APTOS 2019 Blindness Detection dataset for training.

What it does:
  1. Downloads APTOS 2019 via the Kaggle API (needs KAGGLE_USERNAME / KAGGLE_KEY
     or ~/.kaggle/kaggle.json). If credentials are missing it FAILS LOUDLY -
     it will not silently invent data.
  2. Builds a patient-level stratified train/val/test split. APTOS 2019 ships
     one row per image with no patient_id column and, per the competition
     data description, one image per patient - so id_code is used as the
     patient key. If a future drop adds a patient_id column this script picks
     it up automatically and groups by it.
  3. Caches every image resized to CACHE_SIZE x CACHE_SIZE (aspect-preserving,
     padded) so training epochs don't re-decode 3 MB JPEGs.

Fallback: pass --synthetic to generate a small placeholder dataset (noise
images, random labels) so the training loop / Grad-CAM / API can be built and
tested end to end WITHOUT real data. Accuracy numbers from synthetic data are
meaningless - the README says so, and so does the manifest it writes.

Usage:
    python -m src.data.prepare_aptos --out data/aptos
    python -m src.data.prepare_aptos --out data/aptos --synthetic --n 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_SIZE

APTOS_COMPETITION = "aptos2019-blindness-detection"
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Kaggle download
# --------------------------------------------------------------------------- #
def _have_kaggle_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    cfg = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.is_file()


def download_aptos(raw_dir: Path) -> Path:
    if not _have_kaggle_credentials():
        sys.exit(
            "ERROR: Kaggle credentials not found.\n"
            "  Set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json in ~/.kaggle/.\n"
            "  Then accept the competition rules at:\n"
            f"  https://www.kaggle.com/competitions/{APTOS_COMPETITION}/rules\n"
            "  Or run with --synthetic to build a placeholder dataset for pipeline testing."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("ERROR: `kaggle` package not installed. pip install kaggle")

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {APTOS_COMPETITION} to {raw_dir} ...")
    api.competition_download_files(APTOS_COMPETITION, path=str(raw_dir), quiet=False)

    zip_path = raw_dir / f"{APTOS_COMPETITION}.zip"
    if zip_path.is_file():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    train_csv = raw_dir / "train.csv"
    if not train_csv.is_file():
        sys.exit(f"ERROR: expected {train_csv} after extraction; got {list(raw_dir.iterdir())}")
    return raw_dir


# --------------------------------------------------------------------------- #
# Image caching
# --------------------------------------------------------------------------- #
def cache_image(src: Path, dst: Path, size: int) -> bool:
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = img
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), canvas)
    return True


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def patient_level_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str = "diagnosis",
    frac=(0.7, 0.15, 0.15),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign each row a split, keeping all images of a patient together and
    stratifying by the patient's (modal) label."""
    rng = np.random.default_rng(seed)

    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={label_col: "_plabel"})
    )

    split_of: dict = {}
    for lab, grp in per_patient.groupby("_plabel"):
        pids = grp[patient_col].to_numpy()
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(round(n * frac[0]))
        n_va = int(round(n * frac[1]))
        for pid in pids[:n_tr]:
            split_of[pid] = "train"
        for pid in pids[n_tr : n_tr + n_va]:
            split_of[pid] = "val"
        for pid in pids[n_tr + n_va :]:
            split_of[pid] = "test"

    out = df.copy()
    out["split"] = out[patient_col].map(split_of)
    # any stragglers (shouldn't happen) -> train
    out["split"] = out["split"].fillna("train")
    return out


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def build_synthetic(out_dir: Path, n: int) -> None:
    import cv2

    print(f"!!! SYNTHETIC MODE: generating {n} noise images with random labels.")
    print("!!! Any metric computed on this data is meaningless. Train on real APTOS.")
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    # Rough APTOS class imbalance so class-weighting code paths get exercised.
    class_p = np.array([0.49, 0.10, 0.27, 0.05, 0.09])
    for i in range(n):
        label = int(rng.choice(5, p=class_p))
        # a dark frame with a bright off-centre disc + blobs, faintly label-correlated
        img = np.zeros((CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
        cx, cy = CACHE_SIZE // 2, CACHE_SIZE // 2
        cv2.circle(img, (cx, cy), CACHE_SIZE // 2 - 20, (90, 60, 45), -1)
        img = cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))
        for _ in range(label * 6):
            r = int(rng.integers(3, 12))
            px = int(rng.integers(40, CACHE_SIZE - 40))
            py = int(rng.integers(40, CACHE_SIZE - 40))
            cv2.circle(img, (px, py), r, (30, 20, 160), -1)
        name = f"synthetic_{i:05d}.png"
        cv2.imwrite(str(cache / name), img)
        rows.append(
            {
                "id_code": f"synthetic_{i:05d}",
                "patient_id": f"synthetic_{i:05d}",
                "diagnosis": label,
                "image_path": f"cache/{name}",
            }
        )

    df = pd.DataFrame(rows)
    df = patient_level_split(df, patient_col="patient_id")
    _finalise(df, out_dir, synthetic=True)


# --------------------------------------------------------------------------- #
def _finalise(df: pd.DataFrame, out_dir: Path, synthetic: bool) -> None:
    csv_path = out_dir / "labels.csv"
    df.to_csv(csv_path, index=False)

    counts = (
        df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0).to_dict("index")
    )
    manifest = {
        "synthetic": synthetic,
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "splits": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "class_counts_by_split": {k: {int(c): int(v) for c, v in row.items()} for k, row in counts.items()},
        "cache_size": CACHE_SIZE,
        "warning": (
            "SYNTHETIC placeholder data - metrics are meaningless."
            if synthetic
            else "Real APTOS 2019 data."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {csv_path}")
    print(json.dumps(manifest, indent=2))


def build_real(raw_dir: Path, out_dir: Path) -> None:
    import cv2  # noqa: F401  (import here to fail fast if missing)

    df = pd.read_csv(raw_dir / "train.csv")  # columns: id_code, diagnosis
    img_dir = raw_dir / "train_images"

    patient_col = "patient_id" if "patient_id" in df.columns else "id_code"
    if patient_col == "id_code":
        df["patient_id"] = df["id_code"]
        print(
            "Note: APTOS 2019 has no patient_id column and is one image per "
            "patient; using id_code as the patient key."
        )

    cache = out_dir / "cache"
    kept, dropped = [], 0
    for _, row in df.iterrows():
        src = img_dir / f"{row['id_code']}.png"
        if not src.is_file():
            src = img_dir / f"{row['id_code']}.jpg"
        dst = cache / f"{row['id_code']}.png"
        if cache_image(src, dst, CACHE_SIZE):
            kept.append(
                {
                    "id_code": row["id_code"],
                    "patient_id": row["patient_id"],
                    "diagnosis": int(row["diagnosis"]),
                    "image_path": f"cache/{row['id_code']}.png",
                }
            )
        else:
            dropped += 1
    if dropped:
        print(f"Warning: {dropped} images could not be read and were skipped.")

    out = patient_level_split(pd.DataFrame(kept), patient_col="patient_id")
    _finalise(out, out_dir, synthetic=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/aptos", help="output dataset dir")
    ap.add_argument("--raw", default=None, help="dir with an already-downloaded APTOS")
    ap.add_argument("--synthetic", action="store_true", help="build placeholder data")
    ap.add_argument("--n", type=int, default=400, help="synthetic image count")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        build_synthetic(out_dir, args.n)
        return

    raw_dir = Path(args.raw) if args.raw else Path(args.out) / "raw"
    if not (raw_dir / "train.csv").is_file():
        download_aptos(raw_dir)
    build_real(raw_dir, out_dir)


if __name__ == "__main__":
    main()
