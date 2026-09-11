"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _create_test_video(path: Path, num_frames: int = 10, fps: float = 10.0) -> Path:
    w, h = 160, 120
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (w // 2, h // 2), 15, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


class TestPipeline:
    def test_pipeline_imports(self) -> None:
        from mimic.pipeline import run

        assert callable(run)

    def test_pipeline_stages_exist(self) -> None:
        from mimic.export.gltf_writer import write_gltf
        from mimic.extraction.pose_extractor import extract_landmarks
        from mimic.extraction.video_reader import read_frames
        from mimic.processing.rotation_solver import solve_rotations
        from mimic.processing.smoothing import smooth_landmarks
        from mimic.retargeting.retarget import retarget

        assert callable(read_frames)
        assert callable(extract_landmarks)
        assert callable(smooth_landmarks)
        assert callable(solve_rotations)
        assert callable(retarget)
        assert callable(write_gltf)
