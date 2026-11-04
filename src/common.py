"""Shared paths and helpers."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "configs" / "brawl_stars.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "yolov8s_brawl_v2.pt"
RESULTS_DIR = REPO_ROOT / "results"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_data_config(config: str | Path) -> dict:
    """Load a dataset YAML and make its `path` absolute (relative paths start at the repo root)."""
    config = Path(config)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or config.parent)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root} (see README > Dataset)")
    cfg["path"] = root.as_posix()
    return cfg


def resolve_data_yaml(config: str | Path) -> str:
    """Write a copy of the dataset YAML with an absolute `path` and return its location.

    Ultralytics resolves a relative dataset path against its own settings directory, not
    against the YAML file, so the path is made absolute before training or validation.
    """
    cfg = load_data_config(config)
    out = Path(tempfile.gettempdir()) / f"{Path(config).stem}.resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(out)


def split_dir(cfg: dict, split: str) -> Path:
    """Folder that contains the `images/` and `labels/` of a split ('train', 'val' or 'test')."""
    images = Path(cfg["path"]) / cfg[split]  # e.g. <root>/valid/images
    return images.parent


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
