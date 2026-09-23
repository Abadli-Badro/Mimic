"""Paths, constants, and skeleton definitions."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = DATA_DIR / "models"
RIGS_DIR = DATA_DIR / "rigs"

# Pose Landmarker variant: lite (fast, default), full, or heavy.
# Override with MIMIC_POSE_MODEL=heavy|full|lite.
MEDIAPIPE_MODEL_VARIANT = os.environ.get("MIMIC_POSE_MODEL", "lite").strip().lower()
if MEDIAPIPE_MODEL_VARIANT not in {"lite", "full", "heavy"}:
    raise ValueError(
        f"Invalid MIMIC_POSE_MODEL={MEDIAPIPE_MODEL_VARIANT!r}; expected lite, full, or heavy."
    )
MEDIAPIPE_MODEL_PATH = MODELS_DIR / f"pose_landmarker_{MEDIAPIPE_MODEL_VARIANT}.task"

FRAME_WIDTH = 480
FRAME_HEIGHT = 480
TARGET_FPS = 30


def output_file(name: str) -> Path:
    """Resolve a generated artifact under the project output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name

# Resource and detection policy. Limits are checked before and during decoding.
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoLimits:
    max_file_bytes: int = 250 * 1024**2
    max_duration_seconds: float = 120.0
    max_pixels: int = 3840 * 2160
    max_dimension: int = 3840
    max_fps: float = 120.0
    max_decoded_bytes: int = 1024**3
    max_frames: int = 14400


@dataclass(frozen=True)
class PoseQuality:
    visibility_threshold: float = 0.5
    minimum_good_frames: int = 3
    minimum_good_fraction: float = 0.6
    maximum_gap_seconds: float = 1.0


VIDEO_LIMITS = VideoLimits()
POSE_QUALITY = PoseQuality()

# Number of consecutive unobserved source frames allowed before ending a clip.
MAX_MISSING_BONE_FRAMES = 20
# Optional per-bone overrides, using names from rotation_solver.BONE_ORDER.
BONE_MISSING_FRAME_LIMITS = {}
