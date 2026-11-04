"""Object crops cut out of a YOLO-format detection dataset (input of the second-stage classifier)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from common import list_images


def load_crops(split_folder: Path, class_names: list[str], keep: list[str], pad: float = 0.15,
               min_side: int = 4) -> tuple[list[np.ndarray], list[int]]:
    """Cut every box of the `keep` classes out of a split's images.

    Boxes are enlarged by `pad` x their size on each side to keep some context (health bar, shadow...).
    Returns RGB crops and labels re-indexed in the order of `keep`.
    """
    remap = {class_names.index(c): i for i, c in enumerate(keep)}
    crops, labels = [], []
    for img_path in list_images(split_folder / "images"):
        label_path = split_folder / "labels" / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5 or int(parts[0]) not in remap:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            x1, x2 = max(int((xc - bw * (0.5 + pad)) * w), 0), min(int((xc + bw * (0.5 + pad)) * w), w)
            y1, y2 = max(int((yc - bh * (0.5 + pad)) * h), 0), min(int((yc + bh * (0.5 + pad)) * h), h)
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            crops.append(image[y1:y2, x1:x2].copy())
            labels.append(remap[int(parts[0])])
    return crops, labels


class CropDataset(Dataset):
    """In-memory crops with a torchvision transform applied on access."""

    def __init__(self, crops: list[np.ndarray], labels: list[int], transform):
        self.crops, self.labels, self.transform = crops, labels, transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.transform(self.crops[i]), self.labels[i]
