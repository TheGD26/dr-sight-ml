"""PyTorch Dataset + augmentation pipelines for APTOS-style fundus data.

Expects a CSV with columns: id_code, diagnosis, split, [patient_id], image_path
(as produced by prepare_aptos.py). Images are read from the resized cache.

Augmentations are deliberately conservative: rotation, flips, mild
brightness/contrast jitter. No channel swaps, hue shifts, or heavy colour
distortion - those change the clinical appearance of haemorrhages, exudates
and neovascularisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import IMAGENET_MEAN, IMAGENET_STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:  # pragma: no cover - albumentations is a hard dep in practice
    A = None
    ToTensorV2 = None

import cv2


def build_transforms(input_size: int, train: bool):
    """Return an albumentations transform producing a CHW float tensor."""
    if A is None:
        raise ImportError("albumentations is required: pip install albumentations")

    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT),
                A.Affine(
                    rotate=(-25, 25),
                    scale=(0.9, 1.1),
                    translate_percent=(0.0, 0.03),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.12, contrast_limit=0.12, p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class APTOSDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        input_size: int = 300,
        train_aug: bool | None = None,
        data_root: str | Path | None = None,
    ):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows for split={split!r} in {csv_path}")
        self.df = df
        self.split = split
        self.data_root = Path(data_root) if data_root else Path(csv_path).parent
        do_aug = train_aug if train_aug is not None else (split == "train")
        self.tf = build_transforms(input_size, train=do_aug)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row) -> Path:
        p = Path(row["image_path"])
        return p if p.is_absolute() else (self.data_root / p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._resolve_path(row)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tf(image=rgb)["image"]
        label = int(row["diagnosis"])
        return tensor, label

    # --- helpers used by train.py ---
    def labels(self) -> np.ndarray:
        return self.df["diagnosis"].to_numpy(dtype=np.int64)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels(), minlength=5).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(counts) * counts)
        return torch.tensor(w, dtype=torch.float32)
