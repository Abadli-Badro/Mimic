"""OpenCV-based video frame reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(
    video_path: Path,
    target_fps: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Read frames from a video file.

    Args:
        video_path: Path to the input video file.
        target_fps: If provided, subsample to this frame rate.

    Returns:
        A tuple of (frames, effective_fps) where frames is a list of BGR images.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if target_fps and target_fps < source_fps:
        step = max(1, round(source_fps / target_fps))
    else:
        step = 1

    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames, source_fps / step


def get_video_info(video_path: Path) -> dict:
    """Get metadata about a video file.

    Returns:
        Dict with keys: fps, total_frames, duration, width, height.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
