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

    world = result["world_landmarks"]
    vis = result["visibility"]
    landmarks_2d = result["landmarks_2d"]

    # Trim zero frames (MediaPipe returns all-zeros when no pose detected)
    vis_sum = vis.sum(axis=1)
    valid = vis_sum > 1.0
    if not valid.all():
        first_valid = int(np.argmax(valid))
        last_valid = len(valid) - 1 - int(np.argmax(valid[::-1]))
        print(f"Trimming frames 0-{first_valid - 1} and {last_valid + 1}-{len(valid) - 1} "
              f"(no pose detected)")
        world = world[first_valid:last_valid + 1]
        vis = vis[first_valid:last_valid + 1]
        landmarks_2d = landmarks_2d[first_valid:last_valid + 1]
    else:
        first_valid = 0
        last_valid = len(valid) - 1

    np.savez_compressed(
        output_path,
        world_landmarks=world,
        landmarks_2d=landmarks_2d,
        visibility=vis,
        fps=source_fps,
        landmark_names=result["landmark_names"],
        first_frame=first_valid,
    )
    print(f"Saved landmarks to: {output_path}")
    print(f"  Shape: {world.shape} "
          f"({world.shape[0]} frames, "
          f"{world.shape[1]} landmarks, 3 coords)")

    return output_path
