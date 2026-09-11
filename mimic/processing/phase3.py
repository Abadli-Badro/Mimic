"""Phase 3: Solve bone rotations from smoothed landmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.processing.rotation_solver import solve_rotations


def solve(npz_path: Path, output_path: Path | None = None) -> Path:
    """Solve bone rotations from smoothed landmarks.

    Args:
        npz_path: Path to the smoothed .npz file.
        output_path: Path for the output .npz file. Defaults to {name}_rotations.npz.

    Returns:
        Path to the saved rotations .npz file.
    """
    if output_path is None:
        output_path = npz_path.with_name(npz_path.stem.replace("_smooth", "") + "_rotations.npz")

    data = dict(np.load(npz_path, allow_pickle=True))
    world = data["world_landmarks"]
    visibility = data["visibility"]

    print(f"Solving rotations: {world.shape[0]} frames, {world.shape[1]} landmarks")

    result = solve_rotations(world, visibility)

    # Save rotations as individual arrays
    save_dict = {
        "num_frames": result["num_frames"],
        "bone_names": np.array(result["bone_names"]),
    }
    for bone_name, quats in result["rotations"].items():
        save_dict[f"rot_{bone_name}"] = quats

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved rotations to: {output_path}")
    print(f"  {len(result['bone_names'])} bones, {result['num_frames']} frames")
    return output_path
