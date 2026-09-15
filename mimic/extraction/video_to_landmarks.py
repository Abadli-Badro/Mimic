"""Extract landmarks from video and save to .npz."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file
from mimic.errors import stage
from mimic.artifacts import output_path as check_output, save_npz
from mimic.validation import landmarks
from mimic.extraction.quality import validate_pose_quality

import numpy as np

from mimic.extraction.pose_extractor import extract_landmarks
from mimic.extraction.video_reader import get_video_info, read_frames


@stage("extraction")
def extract(video_path: Path, output_path: Path | None = None) -> Path:
    """Extract pose landmarks from a video and save to .npz.

    Args:
        video_path: Path to input video.
        output_path: Path for the output .npz file.

    Returns:
        Path to the saved .npz file.
    """
    if output_path is None:
        output_path = output_file(video_path.stem + ".npz")

    output_path = check_output(output_path, ".npz", [video_path])
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

    landmarks({**result, 'fps': source_fps})
    first_valid, last_valid = validate_pose_quality(world, vis, source_fps)
    if first_valid or last_valid != len(world) - 1:
        print(f"Keeping source frames {first_valid}-{last_valid} (trimming undetected edges)")
    world = world[first_valid:last_valid + 1]
    vis = vis[first_valid:last_valid + 1]
    landmarks_2d = landmarks_2d[first_valid:last_valid + 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
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
