"""Frame sources: a desktop window (Android emulator such as BlueStacks) or an image / video file."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from common import IMG_EXT


class WindowSource:
    """Grab frames from a desktop window (Windows only: relies on PyGetWindow + PyAutoGUI).

    The window is resized and moved to a fixed place so that captures are reproducible, then the
    emulator's title bar (`crop_top` px) and side toolbar (`crop_right` px) are cut off.
    """

    def __init__(self, title: str = "BlueStacks App Player", size=(960, 540), position=(0, 300),
                 crop_top: int = 40, crop_right: int = 41):
        import pygetwindow as gw

        windows = gw.getWindowsWithTitle(title)
        if not windows:
            raise RuntimeError(f"No window whose title contains '{title}'")
        self.window = windows[0]
        self.window.resizeTo(*size)
        self.window.moveTo(*position)
        self.crop_top, self.crop_right = crop_top, crop_right

    def read(self) -> np.ndarray:
        import pyautogui

        w = self.window
        region = (w.left, w.top + self.crop_top, w.width - self.crop_right, w.height - self.crop_top)
        return cv2.cvtColor(np.array(pyautogui.screenshot(region=region)), cv2.COLOR_RGB2BGR)


class FileSource:
    """Frames from an image (returned forever) or a video file (returns None at the end)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.image = cv2.imread(str(self.path)) if self.path.suffix.lower() in IMG_EXT else None
        self.capture = None if self.image is not None else cv2.VideoCapture(str(self.path))
        if self.image is None and not self.capture.isOpened():
            raise FileNotFoundError(self.path)

    def read(self) -> np.ndarray | None:
        if self.image is not None:
            return self.image.copy()
        ok, frame = self.capture.read()
        return frame if ok else None
