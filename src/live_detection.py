"""Real-time object detection on an Android emulator window (BlueStacks) or on an image / video file.

The detector's frame rate is shown in the corner. Press 'q' to quit.

Examples:
    python src/live_detection.py                                  # live, BlueStacks window, GPU 0
    python src/live_detection.py --source clip.mp4 --save demo.mp4
    python src/live_detection.py --source frame.jpg --save pred.jpg --no-show --device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import DEFAULT_WEIGHTS, IMG_EXT
from screen import FileSource, WindowSource

# One distinct BGR colour per class index.
PALETTE = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 0, 128),
           (255, 165, 0), (0, 128, 128), (128, 128, 0), (0, 0, 128), (128, 0, 0), (0, 128, 0), (128, 128, 128)]


def annotate(frame, result, names: dict) -> None:
    """Draw the boxes of one Ultralytics result on `frame` (in place)."""
    boxes = result.boxes
    for (x1, y1, x2, y2), conf, cls in zip(boxes.xyxy.cpu().numpy().astype(int), boxes.conf.cpu().numpy(),
                                           boxes.cls.cpu().numpy().astype(int)):
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", default=None, help="image or video file (default: capture the emulator window)")
    p.add_argument("--window", default="BlueStacks App Player", help="title of the window to capture")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' otherwise")
    p.add_argument("--save", default=None, help="write the annotated output (image or .mp4)")
    p.add_argument("--no-show", action="store_true", help="do not open a display window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until 'q')")
    a = p.parse_args()

    model = YOLO(a.weights)
    source = FileSource(a.source) if a.source else WindowSource(a.window)
    still = isinstance(source, FileSource) and source.image is not None
    save_image = a.save and Path(a.save).suffix.lower() in IMG_EXT
    frame = source.read()
    if frame is not None:  # warm-up: the first call includes model set-up and would distort the FPS
        model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
    writer, fps, n = None, None, 0
    while frame is not None:
        t0 = time.perf_counter()
        result = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        dt = time.perf_counter() - t0
        fps = 1 / dt if fps is None else 0.9 * fps + 0.1 / dt  # exponential moving average
        annotate(frame, result, model.names)
        if not still:  # a frame rate is meaningless on a single image
            cv2.putText(frame, f"detector: {fps:.1f} FPS", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        if a.save and save_image:
            cv2.imwrite(a.save, frame)
        elif a.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("Real-time object detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if (a.max_frames and n >= a.max_frames) or (still and a.no_show):
            break
        frame = source.read()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"{n} frame(s), last detector rate {fps:.1f} FPS" if fps else "no frame processed")


if __name__ == "__main__":
    main()
