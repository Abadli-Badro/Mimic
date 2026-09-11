"""OpenCV-based video frame reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(video_path: Path, target_fps: int | None = None) -> tuple[list[np.ndarray], float]:
    """Read frames from a video file.

    Args:
        video_path: Path to the input video file.
        target_fps: If provided, subsample to this frame rate.

    Returns:
        A tuple of (frames, source_fps) where frames is a list of BGR images.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, source_fps
