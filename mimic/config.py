"""Paths, constants, and skeleton definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RIGS_DIR = DATA_DIR / "rigs"

MEDIAPIPE_MODEL_PATH = MODELS_DIR / "pose_landmarker.task"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
