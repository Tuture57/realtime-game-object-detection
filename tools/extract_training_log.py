"""Recover the per-epoch training log stored inside an Ultralytics checkpoint, without running its pickle code.

Ultralytics stores the training arguments, the best metrics and the results table of the run in every
checkpoint. This script reads them with a restricted unpickler: every class or function reference becomes
an inert stub and tensor storages are never loaded, so even an untrusted .pt file cannot execute code.
It is how the training curves of the released model were recovered after the Colab session was gone.

Example:
    python tools/extract_training_log.py weights/yolov8s_brawl_v2.pt --out results/yolov8s_v2_training
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


_SAFE = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "slice"): slice,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy", "dtype"): np.dtype,
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _SAFE.get((module, name)) or type(name, (_Stub,), {})

    def persistent_load(self, pid):
        return None  # tensor storages are never read


def read_checkpoint_metadata(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        ckpt = _SafeUnpickler(io.BytesIO(z.read(pkl))).load()
    return {k: ckpt.get(k) for k in ("date", "version", "train_args", "train_metrics", "train_results")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True, help="output prefix: writes <out>_log.csv, <out>_summary.json, <out>_curves.png")
    a = p.parse_args()

    meta = read_checkpoint_metadata(a.checkpoint)
    table = meta["train_results"] or {}
    if not table:
        raise SystemExit("no training log in this checkpoint (Ultralytics < 8.0.180?)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = list(table)
    with open(f"{out}_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        writer.writerows(zip(*(table[k] for k in keys)))

    args_keep = ["model", "epochs", "imgsz", "batch", "optimizer", "lr0", "pretrained", "close_mosaic", "seed"]
    summary = {
        "trained_on": meta["date"],
        "ultralytics_version": meta["version"],
        "train_args": {k: (meta["train_args"] or {}).get(k) for k in args_keep},
        "best_epoch_metrics": {k: round(float(v), 4) for k, v in (meta["train_metrics"] or {}).items()},
        "epochs_in_log": len(table.get("epoch", [])),
    }
    Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ep = table["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(ep, table["metrics/mAP50(B)"], label="mAP@0.5")
    ax1.plot(ep, table["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    ax1.set(xlabel="epoch", ylabel="validation mAP", title="Validation accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)
    for k, style in (("box_loss", "-"), ("cls_loss", "--")):
        ax2.plot(ep, table[f"train/{k}"], style, color="tab:blue", label=f"train {k}")
        ax2.plot(ep, table[f"val/{k}"], style, color="tab:orange", label=f"val {k}")
    ax2.set(xlabel="epoch", ylabel="loss", title="Losses")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_curves.png", dpi=150)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}_log.csv, {out}_summary.json, {out}_curves.png")


if __name__ == "__main__":
    main()
