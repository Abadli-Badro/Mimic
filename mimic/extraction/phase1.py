"""Phase 1: Extract landmarks from video and save to .npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.extraction.pose_extractor import extract_landmarks
from mimic.extraction.video_reader import get_video_info, read_frames


def extract(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract pose landmarks from a video and save to .npz.

    Args:
        video_path: Path to input video.
        output_path: Path for the output .npz file.

    Returns:
        Path to the saved .npz file.
    """
    if output_path is None:
        output_path = video_path.with_suffix(".npz")

    info = get_video_info(video_path)
    print(f"Video: {video_path.name} ({info['width']}x{info['height']}, "
          f"{info['fps']:.1f} fps, {info['total_frames']} frames, "
          f"{info['duration']:.1f}s)")

    frames, source_fps = read_frames(video_path)
    print(f"Read {len(frames)} frames (effective fps: {source_fps:.1f})")

    print("Running MediaPipe Pose Landmarker...")
    result = extract_landmarks(frames, fps=source_fps)

    np.savez_compressed(
        output_path,
        world_landmarks=result["world_landmarks"],
        landmarks_2d=result["landmarks_2d"],
        visibility=result["visibility"],
        fps=source_fps,
        landmark_names=result["landmark_names"],
    )
    print(f"Saved landmarks to: {output_path}")
    print(f"  Shape: {result['world_landmarks'].shape} "
          f"({result['world_landmarks'].shape[0]} frames, "
          f"{result['world_landmarks'].shape[1]} landmarks, 3 coords)")

    return output_path
