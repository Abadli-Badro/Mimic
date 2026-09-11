"""End-to-end test for phase 1 extraction pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mimic.extraction.phase1 import extract
from mimic.extraction.video_reader import get_video_info, read_frames


def _create_test_video(path: Path, num_frames: int = 30, fps: float = 10.0) -> Path:
    """Create a simple synthetic test video with a moving circle."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving circle
        cx = int(width * 0.3 + (width * 0.4) * (i / num_frames))
        cy = int(height * 0.5 + 30 * np.sin(2 * np.pi * i / num_frames))
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return path


def test_video_reader():
    """Test basic video reading."""
    video_path = Path("examples/sample_videos/test_synthetic.mp4")
    _create_test_video(video_path)

    info = get_video_info(video_path)
    assert info["fps"] > 0
    assert info["total_frames"] == 30
    assert info["width"] == 320
    assert info["height"] == 240

    frames, fps = read_frames(video_path)
    assert len(frames) == 30
    assert fps > 0

    video_path.unlink(missing_ok=True)


def test_full_extraction():
    """Test full extraction pipeline (video -> .npz)."""
    video_path = Path("examples/sample_videos/test_synthetic.mp4")
    _create_test_video(video_path)

    output_path = Path("examples/sample_videos/test_synthetic.npz")
    result = extract(video_path, output_path)

    assert result.exists()
    data = dict(np.load(result, allow_pickle=True))
    assert "world_landmarks" in data
    assert "visibility" in data
    assert data["world_landmarks"].shape == (30, 33, 3)
    assert data["visibility"].shape == (30, 33)

    result.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_video_reader()
    print("test_video_reader passed")
    test_full_extraction()
    print("test_full_extraction passed")
