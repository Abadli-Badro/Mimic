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
    validate_pose_quality(data["world_landmarks"], data["visibility"], float(data["fps"]))
    world = data["world_landmarks"]
    visibility = data["visibility"]

    print(f"Solving rotations: {world.shape[0]} frames, {world.shape[1]} landmarks")

    result = solve_rotations(world, visibility)

    # Save rotations as individual arrays
    save_dict = {
        "num_frames": result["num_frames"],
        "fps": data["fps"],
        "first_frame": data.get("first_frame", 0),
        "bone_names": np.array(result["bone_names"]),
    }
    for bone_name, quats in result["rotations"].items():
        save_dict[f"rot_{bone_name}"] = quats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, **save_dict)
    print(f"Saved rotations to: {output_path}")
    print(f"  {len(result['bone_names'])} bones, {result['num_frames']} frames")
    return output_path
