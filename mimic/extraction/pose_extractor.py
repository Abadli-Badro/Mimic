"""MediaPipe Pose Landmarker wrapper for per-frame landmark extraction."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mimic.config import FRAME_WIDTH, MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_VARIANT
from mimic.errors import MimicError
from mimic.validation import number
from mimic.extraction.pose_selection import PoseSelector

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

# MediaPipe pose skeleton connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    (0, 1), (1, 2), (2, 3),  # left eye
    (0, 4), (4, 5), (5, 6),  # right eye
    (0, 7), (0, 8),  # ears
    (9, 10),  # mouth
]


def _prefetch(produce, maxsize: int = 2) -> Iterator:
    """Consume produce() on a daemon thread into a bounded queue.

    Overlaps decode/resize/color-convert with MediaPipe inference while
    preserving order. Producer exceptions are re-raised in the consumer.
    """
    items: queue.Queue = queue.Queue(maxsize=maxsize)
    errors: list = []

    def run():
        try:
            for item in produce():
                items.put(item)
        except BaseException as exc:
            errors.append((exc, exc.__traceback__))
        finally:
            items.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def consume():
        while True:
            item = items.get()
            if item is None:
                break
            yield item
        thread.join()
        if errors:
            exc, tb = errors[0]
            raise exc.with_traceback(tb)

    return consume()


def extract_landmarks(
    frames: Iterable[np.ndarray],
    model_path: Path | None = None,
    fps: float | None = None,
) -> dict:
    """Extract 3D pose landmarks from each frame using MediaPipe Pose Landmarker.

    Args:
        frames: Iterable of BGR images (list or generator).
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
            f"pose_landmarker_{MEDIAPIPE_MODEL_VARIANT}/float16/1/"
            f"pose_landmarker_{MEDIAPIPE_MODEL_VARIANT}.task"
        )

    if fps is not None:
        number(fps, 'fps', maximum=240)

    source = iter(frames)
    first = next(source, None)
    if first is None:
        raise MimicError('empty_video', 'No frames were supplied for pose detection.')

    def validated():
        frame, index = first, 0
        while frame is not None:
            if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                    or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size):
                raise MimicError('invalid_video_frame', f'Frame {index} must be a nonempty uint8 BGR image.')
            yield index, frame
            index += 1
            frame = next(source, None)

    base_options = python.BaseOptions(model_asset_path=str(model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
    )

    def prepared():
        for i, frame in validated():
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            if width > FRAME_WIDTH:
                scaled_height = max(1, int(round(height * FRAME_WIDTH / width)))
                frame = cv2.resize(frame, (FRAME_WIDTH, scaled_height),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            yield i, aspect_ratio, rgb

    selector = PoseSelector(fps or 30.0)
    all_world = []
    all_2d = []
    all_visibility = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, aspect_ratio, rgb in _prefetch(prepared):
            timestamp_ms = int((i / fps) * 1000) if fps else i * 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            selected = selector.select(result.pose_landmarks, i, timestamp_ms,
                                       aspect_ratio=aspect_ratio)
            if selected is not None and selected < len(result.pose_world_landmarks):
                lm = result.pose_world_landmarks[selected]
                coords = np.array([[p.x, -p.y, -p.z] for p in lm])
                vis = np.array([p.visibility for p in lm])
            else:
                coords = np.zeros((33, 3))
                vis = np.zeros(33)

            if selected is not None and selected < len(result.pose_world_landmarks):
                lm2d = result.pose_landmarks[selected]
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
