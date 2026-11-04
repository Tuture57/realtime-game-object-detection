"""Collect training screenshots from the emulator window at a fixed interval (Ctrl+C to stop).

The images can then be annotated (e.g. on Roboflow) to grow the dataset.

Example:
    python src/capture.py --interval 1.0 --out data/raw
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from common import REPO_ROOT
from screen import WindowSource


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", default="BlueStacks App Player", help="title of the window to capture")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "raw"))
    p.add_argument("--interval", type=float, default=1.0, help="seconds between two captures")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    source = WindowSource(a.window)
    n = 0
    print(f"Saving a frame every {a.interval}s to {out} (Ctrl+C to stop)")
    try:
        while True:
            cv2.imwrite(str(out / f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"), source.read())
            n += 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print(f"\n{n} screenshot(s) saved")


if __name__ == "__main__":
    main()
