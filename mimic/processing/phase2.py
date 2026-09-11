"""Phase 2: Smooth landmark trajectories from .npz file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.processing.smoothing import smooth_landmarks_array


def smooth(
    npz_path: Path,
    output_path: Path | None = None,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
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
        output_path = npz_path.with_name(npz_path.stem + "_smooth.npz")

    data = dict(np.load(npz_path, allow_pickle=True))
    world = data["world_landmarks"]
    fps = float(data["fps"])

    print(f"Smoothing {world.shape[0]} frames, {world.shape[1]} landmarks")
    print(f"  min_cutoff={min_cutoff}, beta={beta}, fps={fps}")

    smoothed_world = smooth_landmarks_array(
        world, fps=fps, min_cutoff=min_cutoff, beta=beta,
    )

    # 2D landmarks stay as-is (they're already in image space, smoothing would
    # misalign with the video if we did it here)
    np.savez_compressed(
        output_path,
        world_landmarks=smoothed_world,
        landmarks_2d=data["landmarks_2d"],
        visibility=data["visibility"],
        fps=fps,
        landmark_names=data["landmark_names"],
    )
    print(f"Saved smoothed landmarks to: {output_path}")
    return output_path
