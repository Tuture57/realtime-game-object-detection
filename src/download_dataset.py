"""Download the annotated dataset from Roboflow Universe in YOLOv8 format.

The API key is read from the ROBOFLOW_API_KEY environment variable (or asked interactively).
Never hard-code it in a script or a notebook.

Versions:
    1 = 93 annotated screenshots, split 65 / 19 / 9 (train / valid / test), no augmentation
    2 = same images, training set tripled offline (horizontal flip, saturation +-25%)

Example:
    set ROBOFLOW_API_KEY=...        (Windows)   |   export ROBOFLOW_API_KEY=...   (Linux / Colab)
    python src/download_dataset.py --version 1
"""
from __future__ import annotations

import argparse
import getpass
import os

from common import REPO_ROOT

WORKSPACE, PROJECT = "brawlll", "brawl-stars-bot-16ycf"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, default=1, choices=[1, 2])
    p.add_argument("--out", default=None, help="target folder (default: data/brawl-stars-v<version>)")
    a = p.parse_args()

    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY") or getpass.getpass("Roboflow API key: ")
    out = a.out or str(REPO_ROOT / "data" / f"brawl-stars-v{a.version}")
    project = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT)
    project.version(a.version).download("yolov8", location=out)
    print("dataset saved to", out)


if __name__ == "__main__":
    main()
