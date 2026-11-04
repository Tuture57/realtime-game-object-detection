"""Accuracy / latency trade-off of the input resolution for one trained detector.

For every input size: mAP on a split (Ultralytics validator) and end-to-end latency per frame
(pre-processing + inference + NMS, measured around `model.predict`) on that split's images.

Example:
    python src/benchmark.py --split test --imgsz 480 640 800 1024 1280 --device 0
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from common import DEFAULT_DATA, DEFAULT_WEIGHTS, RESULTS_DIR, list_images, load_data_config, split_dir  # noqa: E402
from evaluate import evaluate  # noqa: E402


def cpu_name() -> str:
    """Marketing name of the CPU (Windows registry or /proc/cpuinfo), else the generic platform string."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def device_name(device: str) -> str:
    if device != "cpu" and torch.cuda.is_available():
        return torch.cuda.get_device_name(int(device) if device.isdigit() else 0)
    return cpu_name()


def measure_latency(model: YOLO, frames: list, imgsz: int, device: str, warmup: int = 5, repeats: int = 3) -> list[float]:
    """Wall-clock milliseconds per frame, after a warm-up."""
    sync = (lambda: torch.cuda.synchronize()) if device != "cpu" and torch.cuda.is_available() else (lambda: None)
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(repeats):
        for f in frames:
            sync()
            t0 = time.perf_counter()
            model.predict(f, imgsz=imgsz, device=device, verbose=False)
            sync()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, nargs="+", default=[480, 640, 800, 1024, 1280])
    p.add_argument("--device", default="cpu")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tag", default=None, help="suffix of the output files (default: device type)")
    a = p.parse_args()

    tag = a.tag or ("gpu" if a.device != "cpu" else "cpu")
    frames = [cv2.imread(str(x)) for x in list_images(split_dir(load_data_config(a.data), a.split) / "images")]
    model = YOLO(a.weights)
    hw = device_name(a.device)
    rows = []
    for s in a.imgsz:
        acc = evaluate(a.weights, a.data, a.split, s, a.device)
        t = measure_latency(model, frames, s, a.device, repeats=a.repeats)
        med = statistics.median(t)
        rows.append({
            "imgsz": s, "split": a.split, "mAP50": acc["mAP50"], "mAP50-95": acc["mAP50-95"],
            "latency_ms_mean": round(statistics.mean(t), 1), "latency_ms_median": round(med, 1),
            "latency_ms_p90": round(sorted(t)[int(0.9 * (len(t) - 1))], 1), "fps_median": round(1000 / med, 1),
            "device": hw, "frames": len(t),
        })
        print(rows[-1])

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"resolution_tradeoff_{a.split}_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([r["latency_ms_median"] for r in rows], [r["mAP50-95"] for r in rows], "o-", label="mAP@0.5:0.95")
    ax.plot([r["latency_ms_median"] for r in rows], [r["mAP50"] for r in rows], "s--", label="mAP@0.5")
    for r in rows:
        ax.annotate(f'{r["imgsz"]}px', (r["latency_ms_median"], r["mAP50-95"]), textcoords="offset points",
                    xytext=(4, -12), fontsize=8)
    ax.set_xlabel(f"median latency per frame (ms) - {hw}")
    ax.set_ylabel(f"accuracy on {a.split} split")
    ax.set_title("Input resolution: accuracy vs latency (YOLOv8s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=150)
    print("saved", csv_path, "and", png)


if __name__ == "__main__":
    main()
