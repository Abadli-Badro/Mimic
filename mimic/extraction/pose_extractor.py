"""MediaPipe Pose Landmarker wrapper for per-frame landmark extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.config import MEDIAPIPE_MODEL_PATH


def extract_landmarks(
    frames: list[np.ndarray],
    model_path: Path | None = None,
) -> list[dict[str, np.ndarray]]:
    """Extract 3D pose landmarks from each frame.

    Args:
        frames: List of BGR images.
        model_path: Path to MediaPipe .task model file.

    Returns:
        List of landmark dicts, one per frame. Each dict maps landmark name
        to a numpy array of shape (3,) representing (x, y, z) in normalized
        coordinates.
    """
    _ = model_path or MEDIAPIPE_MODEL_PATH
    # TODO: implement MediaPipe Pose Landmarker inference
    raise NotImplementedError
