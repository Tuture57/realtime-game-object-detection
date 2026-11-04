"""Fine-tune a YOLOv8 detector on the game dataset (thin wrapper around Ultralytics).

Example (settings of the released model, trained on Google Colab):
    python src/train.py --model yolov8s.pt --epochs 50 --imgsz 800 --name yolov8s_800
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from common import DEFAULT_DATA, REPO_ROOT, resolve_data_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained checkpoint (yolov8n/s/m.pt)")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset YAML")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=800, help="training input size (px)")
    p.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    p.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' otherwise")
    p.add_argument("--name", default=None, help="run name under runs/train/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true", help="skip the plots Ultralytics saves after training")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    YOLO(a.model).train(
        data=resolve_data_yaml(a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project=str(REPO_ROOT / "runs" / "train"),
        name=a.name,
        seed=a.seed,
        deterministic=True,
        plots=not a.no_plots,
    )


if __name__ == "__main__":
    main()
