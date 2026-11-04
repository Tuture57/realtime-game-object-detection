"""Plot accuracy against latency for the CPU and GPU runs of src/benchmark.py (log-scale latency axis).

Example:
    python tools/plot_resolution_tradeoff.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def short_name(device: str) -> str:
    cpu = re.search(r"i\d-\w+", device)
    return cpu.group(0) if cpu else device.replace("NVIDIA GeForce ", "")


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for tag, marker in (("gpu", "o"), ("cpu", "s")):
        path = RESULTS / f"resolution_tradeoff_test_{tag}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        x = [float(r["latency_ms_median"]) for r in rows]
        y = [float(r["mAP50-95"]) for r in rows]
        ax.plot(x, y, marker + "-", label=f"{tag.upper()}: {short_name(rows[0]['device'])}")
        for r, xi, yi in zip(rows, x, y):
            ax.annotate(r["imgsz"], (xi, yi), textcoords="offset points", xytext=(3, -11), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("median latency per 1080p frame (ms, log scale)")
    ax.set_ylabel("mAP@0.5:0.95 (test split)")
    ax.set_title("Input resolution (px): accuracy vs latency")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "resolution_tradeoff_test_cpu_gpu.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
