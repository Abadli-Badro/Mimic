"""MediaPipe Pose Landmarker wrapper for per-frame landmark extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mimic.config import MEDIAPIPE_MODEL_PATH

# MediaPipe Pose Landmarker 33 landmark names (index order)
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def extract_landmarks(
    frames: list[np.ndarray],
    model_path: Path | None = None,
    fps: float | None = None,
) -> dict:
    """Extract 3D pose landmarks from each frame using MediaPipe Pose Landmarker.

    Args:
        frames: List of BGR images.
        model_path: Path to MediaPipe .task model file.
        fps: Video frame rate (used for timestamp calculation in VIDEO mode).

    Returns:
        Dict with keys:
          - "world_landmarks": np.ndarray of shape (num_frames, 33, 3) — world coords in meters
          - "visibility": np.ndarray of shape (num_frames, 33) — per-landmark visibility
          - "num_landmarks": int — number of landmarks detected (33)
          - "landmark_names": list[str] — landmark name for each index
    """
    model = model_path or MEDIAPIPE_MODEL_PATH
    if not model.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        )

    base_options = python.BaseOptions(model_asset_path=str(model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    all_world = []
    all_2d = []
    all_visibility = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            timestamp_ms = int((i / fps) * 1000) if fps else i * 33
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                lm = result.pose_world_landmarks[0]
                coords = np.array([[p.x, -p.y, -p.z] for p in lm])
                vis = np.array([p.visibility for p in lm])
            else:
                coords = np.zeros((33, 3))
                vis = np.zeros(33)

            if result.pose_landmarks:
                lm2d = result.pose_landmarks[0]
                coords_2d = np.array([[p.x, p.y] for p in lm2d])
            else:
                coords_2d = np.zeros((33, 2))

            all_world.append(coords)
            all_2d.append(coords_2d)
            all_visibility.append(vis)

    return {
        "world_landmarks": np.array(all_world),
        "landmarks_2d": np.array(all_2d),
        "visibility": np.array(all_visibility),
        "num_landmarks": 33,
        "landmark_names": LANDMARK_NAMES,
    }
