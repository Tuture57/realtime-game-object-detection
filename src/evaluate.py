"""Evaluate a trained detector on one split and save the metrics as JSON.

Example:
    python src/evaluate.py --split test --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from common import DEFAULT_DATA, DEFAULT_WEIGHTS, REPO_ROOT, RESULTS_DIR, resolve_data_yaml


def evaluate(weights: str | Path, data: str | Path = DEFAULT_DATA, split: str = "test", imgsz: int = 800,
             device: str = "cpu", batch: int = 8, plots: bool = False) -> dict:
    """mAP, precision and recall of `weights` on `split`, overall and per class."""
    model = YOLO(str(weights))
    run_name = f"{Path(weights).stem}_{split}_{imgsz}"
    metrics = model.val(data=resolve_data_yaml(data), split=split, imgsz=imgsz, batch=batch, device=device,
                        plots=plots, project=str(REPO_ROOT / "runs" / "val"), name=run_name, exist_ok=True)
    box, names = metrics.box, model.names
    per_class = {
        names[int(c)]: {
            "AP50": round(float(box.ap50[i]), 4),
            "AP50-95": round(float(box.ap[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        }
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "weights": Path(weights).name,
        "split": split,
        "imgsz": imgsz,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
        "device": device,
        "run_dir": str(Path("runs") / "val" / run_name),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plots", action="store_true", help="also save confusion matrix and PR curves")
    p.add_argument("--out", default=None, help="JSON output (default: results/eval_<split>.json)")
    a = p.parse_args()

    res = evaluate(a.weights, a.data, a.split, a.imgsz, a.device, plots=a.plots)
    out = Path(a.out) if a.out else RESULTS_DIR / f"eval_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{a.split}: mAP50={res['mAP50']:.3f}  mAP50-95={res['mAP50-95']:.3f}  "
          f"P={res['precision']:.3f}  R={res['recall']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
