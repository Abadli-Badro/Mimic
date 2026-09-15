"""Intermediate BVH output for debug checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mimic.errors import stage
from mimic.validation import hierarchy as validate_hierarchy, rotations as validate_rotations, number
from mimic.artifacts import atomic_output, output_path as check_output

from mimic.processing.rotation_solver import BONE_HIERARCHY

# BVH channel order
CHANNELS = ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"]
NUM_CHANNELS = 6

# The same rest skeleton as GLB, converted from meters to BVH centimeters.
from mimic.retargeting.skeleton import JOINT_OFFSETS as REST_OFFSETS

JOINT_OFFSETS = {name: tuple(100 * v for v in offset) for name, offset in REST_OFFSETS.items()}


def _get_joint_order() -> list[str]:
    """Channel order must match recursive hierarchy declaration order."""
    return validate_hierarchy(BONE_HIERARCHY)


def _write_joint(f, name: str, offset: tuple, is_root: bool = False) -> None:
    """Write a single joint to the BVH file."""
    if is_root:
        f.write(f"ROOT {name}\n")
    else:
        f.write(f"JOINT {name}\n")
    f.write("{\n")
    f.write(f"\tOFFSET {offset[0]} {offset[1]} {offset[2]}\n")

    has_children = any(p == name for p in BONE_HIERARCHY.values())

    if is_root:
        f.write(f"\tCHANNELS {NUM_CHANNELS} {(' '.join(CHANNELS))}\n")
    else:
        f.write(f"\tCHANNELS {NUM_CHANNELS - 3} Xrotation Yrotation Zrotation\n")

    if not has_children:
        f.write("\tEnd Site\n")
        f.write("{\n")
        f.write("\t\tOFFSET 0 0 0\n")
        f.write("}\n")

    for child, parent in BONE_HIERARCHY.items():
        if parent == name:
            _write_joint(f, child, JOINT_OFFSETS.get(child, (0, 0, 0)))

    f.write("}\n")


@stage("BVH export")
def write_bvh(
    rotations: dict[str, np.ndarray],
    fps: float,
    output_path: Path,
    num_frames: int | None = None,
) -> None:
    """Write bone rotations to a BVH file.

    Args:
        rotations: Dict mapping bone_name -> array of shape (num_frames, 4) quaternions.
        fps: Frame rate.
        output_path: Path to write the .bvh file.
        num_frames: Number of frames. If None, inferred from rotations.
    """
    number(fps, 'fps', maximum=240)
    output_path = check_output(output_path, '.bvh')
    num_frames = validate_rotations(rotations, BONE_HIERARCHY, num_frames)
    joint_order = _get_joint_order()

    with atomic_output(output_path) as temporary, temporary.open('w') as f:
        # Hierarchy
        f.write("HIERARCHY\n")
        _write_joint(f, "pelvis", JOINT_OFFSETS["pelvis"], is_root=True)

        # Motion
        f.write("MOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")

        for frame_idx in range(num_frames):
            values = []
            # Root position (0,0,0) — no root motion
            values.extend([0.0, 0.0, 0.0])
            for joint in joint_order:
                quat = rotations.get(joint, None)
                if quat is None:
                    q = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    q = quat[frame_idx]  # (x, y, z, w)
                # Convert quaternion to Euler angles (BVH uses degrees)
                r = Rotation.from_quat(q)
                euler = r.as_euler("XYZ", degrees=True)
                values.extend(euler.tolist())
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")


from scipy.spatial.transform import Rotation  # noqa: E402
