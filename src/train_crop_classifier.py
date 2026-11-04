"""Second-stage crop classifier, trained with a hand-written PyTorch loop (no Ultralytics).

Question: given a correct box, how separable are the object classes (player / ally / enemy, bush / wall...)?
Comparing its confusion matrix with the detector's shows whether the detector's mistakes come from
classification or from localisation and missed objects.

Examples:
    python src/train_crop_classifier.py --arch resnet18             # ImageNet-pretrained ResNet-18
    python src/train_crop_classifier.py --arch smallcnn             # small CNN trained from scratch
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from common import DEFAULT_DATA, REPO_ROOT, RESULTS_DIR, load_data_config, split_dir
from crop_dataset import CropDataset, load_crops

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Classes with enough boxes (19+); the 5 UI / marker classes have a single example each.
KEEP = ["ally", "baril", "buisson", "crate", "ennemi", "gemme", "generateur", "player", "wall"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(arch: str, n_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    """Return the model and whether ImageNet weights were actually loaded."""
    if arch == "resnet18":
        model, loaded = None, False
        if pretrained:
            try:
                model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
                loaded = True
            except Exception as exc:  # offline machine, blocked download...
                print(f"! ImageNet weights unavailable ({exc}); training from scratch")
        model = model or torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model, loaded

    def block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    model = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))
    return model, False


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns mean loss, predictions, targets."""
    training = optimizer is not None
    model.train(training)
    total, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    targets = torch.cat(targets).numpy()
    return total / len(targets), torch.cat(preds).numpy(), targets


def confusion(y_true, y_pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def scores(cm: np.ndarray) -> dict:
    tp, support, predicted = np.diag(cm).astype(float), cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1[support > 0].mean()),
            "recall": recall, "support": support}


def plot_confusion(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "smallcnn"])
    p.add_argument("--no-pretrained", action="store_true", help="ResNet-18 from random weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img", type=int, default=96, help="crop size fed to the network (px)")
    p.add_argument("--pad", type=float, default=0.15, help="context added around each box")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cfg = load_data_config(a.data)
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    data = {s: load_crops(split_dir(cfg, s), names, KEEP, a.pad) for s in ("train", "val", "test")}
    for s, (_, y) in data.items():
        print(s, len(y), "crops", dict(zip(KEEP, np.bincount(y, minlength=len(KEEP)).tolist())))

    # No hue jitter: team colours (health bars) are one of the cues that separate player / ally / enemy.
    train_tf = T.Compose([T.ToPILImage(), T.RandomResizedCrop(a.img, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToPILImage(), T.Resize((a.img, a.img)), T.ToTensor(), T.Normalize(MEAN, STD)])
    loaders = {s: DataLoader(CropDataset(*data[s], train_tf if s == "train" else eval_tf), batch_size=a.batch,
                             shuffle=s == "train") for s in data}

    model, pretrained = build_model(a.arch, len(KEEP), a.arch == "resnet18" and not a.no_pretrained)
    model.to(device)
    counts = np.bincount(data["train"][1], minlength=len(KEEP))
    weights = torch.tensor(counts.sum() / (len(KEEP) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)  # inverse-frequency class weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_t = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_p, va_t = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        va = scores(confusion(va_t, va_p, len(KEEP)))
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(float((tr_p == tr_t).mean()), 4),
                        "val_loss": round(va_loss, 4), "val_acc": round(va["accuracy"], 4),
                        "val_macro_f1": round(va["macro_f1"], 4), "seconds": round(time.time() - t0, 1)})
        print(history[-1])
        if va["macro_f1"] > best_f1:  # model selection on the validation split only
            best_f1, best_state, best_epoch = va["macro_f1"], copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    tag = a.arch + ("" if a.arch == "smallcnn" else ("_imagenet" if pretrained else "_scratch"))
    out = RESULTS_DIR / "crop_classifier"
    out.mkdir(parents=True, exist_ok=True)
    results = {"model": tag, "imagenet_pretrained": pretrained, "best_epoch": best_epoch, "epochs": a.epochs,
               "img": a.img, "pad": a.pad, "seed": a.seed, "device": device, "classes": KEEP,
               "crops": {s: len(data[s][1]) for s in data}}
    for split in ("val", "test"):
        _, pred, true = run_epoch(model, loaders[split], criterion, device)
        cm = confusion(true, pred, len(KEEP))
        sc = scores(cm)
        results[split] = {"accuracy": round(sc["accuracy"], 4), "macro_f1": round(sc["macro_f1"], 4),
                          "per_class_recall": {c: round(float(r), 3) for c, r in zip(KEEP, sc["recall"])},
                          "support": {c: int(s) for c, s in zip(KEEP, sc["support"])},
                          "confusion_matrix": cm.tolist()}
        plot_confusion(cm, KEEP, f"Crop classifier ({tag}) - {split} split", out / f"{tag}_confusion_{split}.png")
        print(f"{split}: accuracy={sc['accuracy']:.3f} macro-F1={sc['macro_f1']:.3f}")

    (out / f"{tag}_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out / f"{tag}_history.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    ckpt_dir = REPO_ROOT / "runs" / "crop_classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": KEEP, "arch": a.arch, "img": a.img},
               ckpt_dir / f"{tag}.pt")
    print("saved", out / f"{tag}_metrics.json")


if __name__ == "__main__":
    main()
