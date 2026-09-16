"""Solve bone rotations from smoothed landmarks."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file
from mimic.errors import stage
from mimic.artifacts import load_npz, save_npz, output_path as check_output
from mimic.validation import landmarks
from mimic.extraction.quality import validate_pose_quality

import numpy as np

from mimic.processing.rotation_solver import solve_rotations


@stage("rotation solving")
def solve(npz_path: Path, output_path: Path | None = None) -> Path:
    """Solve bone rotations from smoothed landmarks.

    Args:
        npz_path: Path to the smoothed .npz file.
        output_path: Path for the output .npz file. Defaults to {name}_rotations.npz.

    Returns:
        Path to the saved rotations .npz file.
    """
    if output_path is None:
        output_path = output_file(npz_path.stem.removesuffix("_smooth") + "_rotations.npz")

    output_path = check_output(output_path, ".npz", [npz_path])
    data = load_npz(npz_path)
    landmarks(data)

    world = data["world_landmarks"]
    visibility = data["visibility"]

    print(f"Solving rotations: {world.shape[0]} frames, {world.shape[1]} landmarks")

    from mimic.config import MAX_MISSING_BONE_FRAMES
    timer = int(data.get('max_missing_frames', MAX_MISSING_BONE_FRAMES))
    overrides = dict(zip(data.get('bone_timer_names', []), data.get('bone_timer_limits', [])))
    result = solve_rotations(world, visibility, max_missing_frames=timer, bone_limits=overrides)
    stopped = list(result['stopped_bones']) or list(data.get('stopped_bones', []))
    if stopped:
        print(f"Clip ended after {result['num_frames']} frames; expired bones: {', '.join(stopped)}")

    # Save rotations as individual arrays
    save_dict = {
        "num_frames": result["num_frames"],
        "fps": data["fps"],
        "first_frame": data.get("first_frame", 0),
        "bone_names": np.array(result["bone_names"]),
        "input_frames": data.get('input_frames', result['input_frames']),
        "stopped_bones": np.array(stopped, dtype=str),
        "max_missing_frames": timer,
    }
    for bone_name, quats in result["rotations"].items():
        save_dict[f"rot_{bone_name}"] = quats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, **save_dict)
    print(f"Saved rotations to: {output_path}")
    print(f"  {len(result['bone_names'])} bones, {result['num_frames']} frames")
    return output_path
