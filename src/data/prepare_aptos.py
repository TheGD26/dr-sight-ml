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
    # official competition source (needs phone-verified account + accepted rules)
    python -m src.data.prepare_aptos --out data/aptos

    # if the Competitions API 401s: pull a mirror via the Datasets API instead
    python -m src.data.prepare_aptos --out data/aptos --kaggle-dataset mariaherrerot/aptos2019

    # already downloaded a copy yourself? point at it (train.csv + image files)
    python -m src.data.prepare_aptos --out data/aptos --raw /path/to/unzipped

    # no Kaggle access at all: placeholder data for pipeline testing only
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


_CRED_HELP = (
    "  Set KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json in ~/.kaggle/ (chmod 600).\n"
    f"  Then accept the competition rules at:\n"
    f"  https://www.kaggle.com/competitions/{APTOS_COMPETITION}/rules\n"
    "  Or run with --synthetic to build a placeholder dataset for pipeline testing."
)


def _authed_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("ERROR: `kaggle` package not installed. pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    return api


def _extract_all_zips(raw_dir: Path) -> None:
    """Kaggle sometimes delivers one outer zip, sometimes per-file zips.
    Extract every zip we find, including nested ones, then drop the archives."""
    for _ in range(3):  # a couple of nesting levels is plenty
        zips = list(raw_dir.rglob("*.zip"))
        if not zips:
            break
        for zp in zips:
            print(f"Extracting {zp.relative_to(raw_dir)} ...")
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(zp.parent)
            zp.unlink()


def download_aptos(raw_dir: Path) -> Path:
    """Download the official APTOS 2019 *competition* files.

    Needs: valid Kaggle token + a phone-verified account + accepted competition
    rules. If the Competitions API returns 401/403 (common on new accounts even
    after phone verification, or when the site Terms need re-accepting in a
    browser), fall back to `--kaggle-dataset <owner/slug>` which uses the
    Datasets API instead. See README / DEPLOY notes.
    """
    if not _have_kaggle_credentials():
        sys.exit("ERROR: Kaggle credentials not found.\n" + _CRED_HELP)

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = _authed_api()
    print(f"Downloading competition '{APTOS_COMPETITION}' to {raw_dir} ...")
    try:
        api.competition_download_files(APTOS_COMPETITION, path=str(raw_dir), quiet=False)
    except Exception as e:  # noqa: BLE001 - want the friendly hint on any failure
        msg = str(e)
        hint = ""
        if "401" in msg or "403" in msg or "Unauthorized" in msg or "Forbidden" in msg:
            hint = (
                "\n\nThe Competitions API rejected the request (401/403). This is\n"
                "usually one of:\n"
                "  * competition rules not accepted -> open the /rules page, click\n"
                "    'Join Competition' / 'I Understand and Accept';\n"
                "  * account needs to re-accept Kaggle's site Terms -> log in at\n"
                "    kaggle.com in a browser and clear any Terms banner;\n"
                "  * a stale API token -> Settings -> API -> 'Create New Token',\n"
                "    replace ~/.kaggle/kaggle.json, retry.\n"
                "Workaround that does NOT touch the Competitions API:\n"
                "  python -m src.data.prepare_aptos --out data/aptos \\\n"
                "      --kaggle-dataset mariaherrerot/aptos2019\n"
            )
        sys.exit(f"ERROR: competition download failed: {e}{hint}")

    _extract_all_zips(raw_dir)
    if _find_label_csv(raw_dir) is None:
        sys.exit(
            f"ERROR: no APTOS label CSV found under {raw_dir} after extraction; "
            f"got {[p.name for p in raw_dir.iterdir()]}"
        )
    return raw_dir


def download_aptos_dataset(slug: str, raw_dir: Path) -> Path:
    """Download an APTOS 2019 *dataset* mirror via the Datasets API.

    The Datasets API is not gated behind competition-rule acceptance, so this
    works with a plain valid token when `download_aptos()` hits a 401. The
    trade-off: `slug` is a community re-upload, not the official competition
    source - note that in your write-up. A commonly-complete mirror is
    'mariaherrerot/aptos2019'.
    """
    if not _have_kaggle_credentials():
        sys.exit("ERROR: Kaggle credentials not found.\n" + _CRED_HELP)

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = _authed_api()
    print(f"Downloading dataset '{slug}' to {raw_dir} (Datasets API) ...")
    try:
        api.dataset_download_files(slug, path=str(raw_dir), quiet=False, unzip=True)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"ERROR: dataset download failed for '{slug}': {e}\n"
            "  Check the slug exists (kaggle.com/datasets/<slug>) and that your\n"
            "  token is valid. Try a different mirror if this one was removed."
        )
    _extract_all_zips(raw_dir)
    if _find_label_csv(raw_dir) is None:
        sys.exit(
            f"ERROR: '{slug}' downloaded but no train/label CSV with id+diagnosis "
            f"columns was found under {raw_dir}. Try --raw with a manual download, "
            "or a different mirror slug."
        )
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


# Column-name variants seen across the official set and community mirrors.
_ID_COLS = ("id_code", "image", "image_id", "id", "id_code ")
_LABEL_COLS = ("diagnosis", "level", "label", "dr", "grade")
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _find_label_csv(raw_dir: Path) -> tuple[Path, str, str] | None:
    """Locate the APTOS *train* label CSV anywhere under raw_dir.

    Returns (csv_path, id_col, label_col) or None. Prefers a file literally
    named train.csv; otherwise the matching CSV with the most rows, and never
    an obvious test-only CSV (no label column).
    """
    best = None  # (score, rows, path, id_col, label_col)
    for csv in raw_dir.rglob("*.csv"):
        try:
            head = pd.read_csv(csv, nrows=5)
        except Exception:
            continue
        cols = {c.lower().strip(): c for c in head.columns}
        id_col = next((cols[c] for c in _ID_COLS if c in cols), None)
        label_col = next((cols[c] for c in _LABEL_COLS if c in cols), None)
        if id_col is None or label_col is None:
            continue
        try:
            rows = sum(1 for _ in open(csv, "rb")) - 1
        except Exception:
            rows = len(head)
        name = csv.name.lower()
        score = 0
        if name == "train.csv":
            score += 100
        if "train" in name:
            score += 10
        if "test" in name:
            score -= 50
        cand = (score, rows, csv, id_col, label_col)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None
    _, _, path, id_col, label_col = best
    return path, id_col, label_col


def _build_image_index(raw_dir: Path) -> dict[str, Path]:
    """stem -> path for every image under raw_dir. First match wins; prefer
    paths containing 'train' so a combined 2015+2019 mirror doesn't grab the
    2015 test images."""
    index: dict[str, Path] = {}
    for p in raw_dir.rglob("*"):
        if p.suffix not in _IMG_EXTS or not p.is_file():
            continue
        stem = p.stem
        if stem not in index or ("train" in str(p).lower() and "train" not in str(index[stem]).lower()):
            index[stem] = p
    return index


def build_real(raw_dir: Path, out_dir: Path) -> None:
    import cv2  # noqa: F401  (import here to fail fast if missing)

    found = _find_label_csv(raw_dir)
    if found is None:
        sys.exit(f"ERROR: no usable label CSV under {raw_dir}")
    csv_path, id_col, label_col = found
    df = pd.read_csv(csv_path)
    print(f"Labels: {csv_path.relative_to(raw_dir)}  (id='{id_col}', label='{label_col}', {len(df)} rows)")

    img_index = _build_image_index(raw_dir)
    print(f"Indexed {len(img_index)} image files under {raw_dir}")

    patient_col = "patient_id" if "patient_id" in df.columns else None
    if patient_col is None:
        df["patient_id"] = df[id_col].astype(str)
        patient_col = "patient_id"
        print(
            "Note: APTOS 2019 has no patient_id column and is one image per "
            f"patient; using '{id_col}' as the patient key."
        )

    cache = out_dir / "cache"
    kept, dropped = [], 0
    for _, row in df.iterrows():
        code = str(row[id_col])
        src = img_index.get(code) or img_index.get(Path(code).stem)
        dst = cache / f"{code}.png"
        if src is not None and cache_image(src, dst, CACHE_SIZE):
            kept.append(
                {
                    "id_code": code,
                    "patient_id": str(row[patient_col]),
                    "diagnosis": int(row[label_col]),
                    "image_path": f"cache/{code}.png",
                }
            )
        else:
            dropped += 1
    if dropped:
        print(f"Warning: {dropped}/{len(df)} rows had no readable image and were skipped.")
    if not kept:
        sys.exit(
            "ERROR: 0 images matched the label CSV. The mirror's image filenames "
            "probably don't match the CSV id column. Inspect the download and use "
            "--raw after arranging it as train.csv + image files."
        )

    out = patient_level_split(pd.DataFrame(kept), patient_col="patient_id")
    _finalise(out, out_dir, synthetic=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/aptos", help="output dataset dir")
    ap.add_argument("--raw", default=None, help="dir with an already-downloaded APTOS")
    ap.add_argument(
        "--kaggle-dataset",
        default=None,
        metavar="OWNER/SLUG",
        help="pull an APTOS mirror via the Datasets API instead of the "
        "Competitions API (use when the competition download 401s). "
        "e.g. mariaherrerot/aptos2019",
    )
    ap.add_argument("--synthetic", action="store_true", help="build placeholder data")
    ap.add_argument("--n", type=int, default=400, help="synthetic image count")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        build_synthetic(out_dir, args.n)
        return

    raw_dir = Path(args.raw) if args.raw else out_dir / "raw"
    if args.raw is None and _find_label_csv(raw_dir) is None:
        if args.kaggle_dataset:
            download_aptos_dataset(args.kaggle_dataset, raw_dir)
        else:
            download_aptos(raw_dir)
    build_real(raw_dir, out_dir)


if __name__ == "__main__":
    main()
