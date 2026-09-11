"""Tests for extraction modules (video_reader, pose_extractor, visualize)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mimic.extraction.pose_extractor import LANDMARK_NAMES
from mimic.extraction.video_reader import get_video_info, read_frames


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a synthetic test video and return its path."""
    path = tmp_path / "test.mp4"
    w, h = 320, 240
    fps = 24.0
    num_frames = 20
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = int(w * 0.3 + (w * 0.4) * (i / num_frames))
        cy = int(h * 0.5 + 30 * np.sin(2 * np.pi * i / num_frames))
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


class TestVideoReader:
    def test_read_frames_returns_all(self, synthetic_video: Path) -> None:
        frames, fps = read_frames(synthetic_video)
        assert len(frames) == 20
        assert fps == pytest.approx(24.0)
        assert frames[0].shape == (240, 320, 3)

    def test_read_frames_with_subsampling(self, synthetic_video: Path) -> None:
        frames, fps = read_frames(synthetic_video, target_fps=12)
        assert len(frames) == 10
        assert fps == pytest.approx(24.0)

    def test_read_frames_no_subsampling_when_target_higher(self, synthetic_video: Path) -> None:
        frames, _ = read_frames(synthetic_video, target_fps=48)
        assert len(frames) == 20

    def test_read_frames_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_frames(tmp_path / "nonexistent.mp4")

    def test_get_video_info(self, synthetic_video: Path) -> None:
        info = get_video_info(synthetic_video)
        assert info["fps"] == pytest.approx(24.0)
        assert info["total_frames"] == 20
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["duration"] == pytest.approx(20 / 24.0)

    def test_get_video_info_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_video_info(tmp_path / "nonexistent.mp4")


class TestPoseExtractor:
    def test_landmark_names_count(self) -> None:
        assert len(LANDMARK_NAMES) == 33

    def test_landmark_names_are_strings(self) -> None:
        assert all(isinstance(n, str) for n in LANDMARK_NAMES)

    def test_extract_landmarks_missing_model(self, synthetic_video: Path, tmp_path: Path) -> None:
        frames, _ = read_frames(synthetic_video)
        with pytest.raises(FileNotFoundError, match="MediaPipe model not found"):
            from mimic.extraction.pose_extractor import extract_landmarks

            extract_landmarks(frames, model_path=tmp_path / "missing.task")

    @pytest.mark.skipif(
        not Path("data/models/pose_landmarker.task").exists(),
        reason="MediaPipe model not downloaded",
    )
    def test_extract_landmarks_output_shapes(self, synthetic_video: Path) -> None:
        from mimic.extraction.pose_extractor import extract_landmarks

        frames, fps = read_frames(synthetic_video)
        result = extract_landmarks(frames, fps=fps)

        assert result["world_landmarks"].shape == (20, 33, 3)
        assert result["visibility"].shape == (20, 33)
        assert result["num_landmarks"] == 33
        assert len(result["landmark_names"]) == 33


class TestVisualize:
    def test_plot_landmarks_saves_file(self, tmp_path: Path) -> None:
        from mimic.extraction.visualize import plot_landmarks_3d

        landmarks = np.random.rand(5, 33, 3).astype(np.float32)
        out = tmp_path / "test_plot.png"
        plot_landmarks_3d(landmarks, frame_idx=0, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_all_frames(self, tmp_path: Path) -> None:
        from mimic.extraction.visualize import plot_landmarks_all_frames

        landmarks = np.random.rand(20, 33, 3).astype(np.float32)
        vis_dir = tmp_path / "frames"
        saved = plot_landmarks_all_frames(landmarks, vis_dir, max_frames=5)
        assert len(saved) == 5
        for p in saved:
            assert p.exists()
