"""Smooth landmark trajectories from .npz file."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file, MAX_MISSING_BONE_FRAMES
from mimic.processing.tracking import tracking_window
from mimic.errors import stage
from mimic.artifacts import load_npz, save_npz, output_path as check_output
from mimic.validation import landmarks
from mimic.extraction.quality import validate_pose_quality

import numpy as np

from mimic.processing.smoothing import fill_landmark_gaps, smooth_landmarks_array


@stage("smoothing")
def smooth(
    npz_path: Path,
    output_path: Path | None = None,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    max_missing_frames: int = MAX_MISSING_BONE_FRAMES,
    bone_limits: dict[str, int] | None = None,
) -> Path:
    """Smooth landmarks in a .npz file and save the result.

    Args:
        npz_path: Path to the input .npz file.
        output_path: Path for the output .npz file. Defaults to {name}_smooth.npz.
        min_cutoff: Minimum cutoff frequency for One-Euro filter.
        beta: Speed coefficient for One-Euro filter.

    Returns:
        Path to the smoothed .npz file.
    """
    if output_path is None:
        output_path = output_file(npz_path.stem + "_smooth.npz")

    output_path = check_output(output_path, ".npz", [npz_path])
    data = load_npz(npz_path)
    landmarks(data)

    window = tracking_window(data['world_landmarks'], data['visibility'],
                             max_missing_frames=max_missing_frames, bone_limits=bone_limits)
    count = window['num_frames']
    world = data['world_landmarks'][:count]
    visibility = data['visibility'][:count]
    if window['stopped_bones']:
        source_frame = int(data.get('first_frame', 0)) + count
        print(f"Tracking timeout at source frame {source_frame}: {', '.join(window['stopped_bones'])}. "
              f"Keeping {count} frames.")
    fps = float(data["fps"])

    print(f"Smoothing {world.shape[0]} frames, {world.shape[1]} landmarks")
    print(f"  min_cutoff={min_cutoff}, beta={beta}, fps={fps}")

    smoothed_world = smooth_landmarks_array(
        fill_landmark_gaps(world, visibility), fps=fps, min_cutoff=min_cutoff, beta=beta,
    )

    # 2D landmarks stay as-is (they're already in image space, smoothing would
    # misalign with the video if we did it here)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        output_path,
        world_landmarks=smoothed_world,
        landmarks_2d=data["landmarks_2d"][:count],
        visibility=visibility,
        fps=fps,
        landmark_names=data["landmark_names"],
        first_frame=data.get("first_frame", 0),
        input_frames=window['input_frames'],
        stopped_bones=np.array(window['stopped_bones'], dtype=str),
        max_missing_frames=max_missing_frames,
        bone_timer_names=np.array(list(window['limits'])),
        bone_timer_limits=np.array(list(window['limits'].values())),
    )
    print(f"Saved smoothed landmarks to: {output_path}")
    return output_path
