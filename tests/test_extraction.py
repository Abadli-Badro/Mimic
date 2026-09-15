"""Tests for extraction modules (video_reader, pose_extractor, visualize, overlay)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mimic.extraction.pose_extractor import LANDMARK_NAMES, POSE_CONNECTIONS
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
        assert fps == pytest.approx(12.0)

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
        from mimic.extraction.landmark_plots import plot_landmarks_3d

        landmarks = np.random.rand(5, 33, 3).astype(np.float32)
        out = tmp_path / "test_plot.png"
        plot_landmarks_3d(landmarks, frame_idx=0, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_all_frames(self, tmp_path: Path) -> None:
        from mimic.extraction.landmark_plots import plot_landmarks_all_frames

        landmarks = np.random.rand(20, 33, 3).astype(np.float32)
        vis_dir = tmp_path / "frames"
        saved = plot_landmarks_all_frames(landmarks, vis_dir, max_frames=5)
        assert len(saved) == 5
        for p in saved:
            assert p.exists()


class TestPoseConnections:
    def test_pose_connections_count(self) -> None:
        assert len(POSE_CONNECTIONS) > 0

    def test_pose_connections_are_pairs(self) -> None:
        for conn in POSE_CONNECTIONS:
            assert len(conn) == 2

    def test_pose_connections_valid_indices(self) -> None:
        for i, j in POSE_CONNECTIONS:
            assert 0 <= i < 33
            assert 0 <= j < 33


class TestOverlay:
    def test_draw_skeleton_returns_frame(self) -> None:
        from mimic.extraction.overlay import draw_skeleton

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks_2d = np.random.rand(33, 2).astype(np.float32)
        visibility = np.ones(33, dtype=np.float32)
        result = draw_skeleton(frame, landmarks_2d, visibility, 320, 240)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_draw_skeleton_does_not_modify_original(self) -> None:
        from mimic.extraction.overlay import draw_skeleton

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original = frame.copy()
        landmarks_2d = np.random.rand(33, 2).astype(np.float32)
        visibility = np.ones(33, dtype=np.float32)
        draw_skeleton(frame, landmarks_2d, visibility, 320, 240)
        np.testing.assert_array_equal(frame, original)

    def test_draw_skeleton_respects_threshold(self) -> None:
        from mimic.extraction.overlay import draw_skeleton

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        landmarks_2d = np.random.rand(33, 2).astype(np.float32)
        visibility = np.zeros(33, dtype=np.float32)
        result_low = draw_skeleton(frame, landmarks_2d, visibility, 320, 240, threshold=0.5)
        result_high = draw_skeleton(frame, landmarks_2d, visibility, 320, 240, threshold=0.0)
        # With all-zero visibility and threshold=0.5, no landmarks drawn
        # With threshold=0.0, all landmarks drawn
        assert not np.array_equal(result_low, result_high)

    def test_generate_overlay_video(self, tmp_path: Path) -> None:
        from mimic.extraction.overlay import generate_overlay_video

        # Create test video
        video_path = tmp_path / "test.mp4"
        w, h = 160, 120
        fps = 10.0
        num_frames = 5
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Create matching .npz
        npz_path = tmp_path / "test.npz"
        landmarks_2d = np.random.rand(num_frames, 33, 2).astype(np.float32)
        visibility = np.ones((num_frames, 33), dtype=np.float32)
        np.savez_compressed(npz_path, landmarks_2d=landmarks_2d, visibility=visibility)

        out_path = tmp_path / "overlay.mp4"
        result = generate_overlay_video(video_path, npz_path, out_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_generate_overlay_video_no_output_defaults_name(self, tmp_path: Path) -> None:
        from mimic.extraction.overlay import generate_overlay_video

        video_path = tmp_path / "clip.mp4"
        w, h = 160, 120
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        for _ in range(3):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

        npz_path = tmp_path / "clip.npz"
        np.savez_compressed(
            npz_path,
            landmarks_2d=np.zeros((3, 33, 2), dtype=np.float32),
            visibility=np.ones((3, 33), dtype=np.float32),
        )

        result = generate_overlay_video(video_path, npz_path)
        assert result.name == "clip_overlay.mp4"
        assert result.exists()

    def test_generate_overlay_video_handles_frame_mismatch(self, tmp_path: Path) -> None:
        from mimic.extraction.overlay import generate_overlay_video

        video_path = tmp_path / "short.mp4"
        w, h = 80, 60
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        for _ in range(10):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

        npz_path = tmp_path / "short.npz"
        np.savez_compressed(
            npz_path,
            landmarks_2d=np.zeros((3, 33, 2), dtype=np.float32),
            visibility=np.ones((3, 33), dtype=np.float32),
        )

        out_path = tmp_path / "out.mp4"
        result = generate_overlay_video(video_path, npz_path, out_path)
        assert result.exists()
