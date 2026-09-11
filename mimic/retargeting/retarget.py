"""Retarget computed rotations onto a target Mixamo skeleton."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAPPING_PATH = Path(__file__).parent / "mixamo_mapping.json"

# Mapping from internal bone names to Mixamo bone names
INTERNAL_TO_MIXAMO = {
    "pelvis": "mixamorig:Hips",
    "spine": "mixamorig:Spine",
    "chest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    "left_upper_arm": "mixamorig:LeftArm",
    "left_lower_arm": "mixamorig:LeftForeArm",
    "left_hand": "mixamorig:LeftHand",
    "right_upper_arm": "mixamorig:RightArm",
    "right_lower_arm": "mixamorig:RightForeArm",
    "right_hand": "mixamorig:RightHand",
    "left_upper_leg": "mixamorig:LeftUpLeg",
    "left_lower_leg": "mixamorig:LeftLeg",
    "left_foot": "mixamorig:LeftFoot",
    "right_upper_leg": "mixamorig:RightUpLeg",
    "right_lower_leg": "mixamorig:RightLeg",
    "right_foot": "mixamorig:RightFoot",
}

# Mixamo skeleton hierarchy
MIXAMO_HIERARCHY = {
    "mixamorig:Hips": None,
    "mixamorig:Spine": "mixamorig:Hips",
    "mixamorig:Spine1": "mixamorig:Spine",
    "mixamorig:Spine2": "mixamorig:Spine1",
    "mixamorig:Neck": "mixamorig:Spine2",
    "mixamorig:Head": "mixamorig:Neck",
    "mixamorig:LeftShoulder": "mixamorig:Spine2",
    "mixamorig:LeftArm": "mixamorig:LeftShoulder",
    "mixamorig:LeftForeArm": "mixamorig:LeftArm",
    "mixamorig:LeftHand": "mixamorig:LeftForeArm",
    "mixamorig:RightShoulder": "mixamorig:Spine2",
    "mixamorig:RightArm": "mixamorig:RightShoulder",
    "mixamorig:RightForeArm": "mixamorig:RightArm",
    "mixamorig:RightHand": "mixamorig:RightForeArm",
    "mixamorig:LeftUpLeg": "mixamorig:Hips",
    "mixamorig:LeftLeg": "mixamorig:LeftUpLeg",
    "mixamorig:LeftFoot": "mixamorig:LeftLeg",
    "mixamorig:RightUpLeg": "mixamorig:Hips",
    "mixamorig:RightLeg": "mixamorig:RightUpLeg",
    "mixamorig:RightFoot": "mixamorig:RightLeg",
}


def load_mapping() -> dict[str, str]:
    """Load the internal-bone-to-Mixamo-bone name mapping."""
    return INTERNAL_TO_MIXAMO.copy()


def retarget(
    rotations: dict[str, np.ndarray],
    num_frames: int | None = None,
) -> dict:
    """Map internal bone rotations onto the Mixamo skeleton.

    Args:
        rotations: Dict mapping bone_name -> (num_frames, 4) quaternions.
        num_frames: Number of frames.

    Returns:
        Dict with keys:
          - "rotations": dict mapping Mixamo bone name -> (num_frames, 4) quaternions
          - "bone_names": list of Mixamo bone names
          - "hierarchy": dict mapping bone -> parent bone
          - "num_frames": int
    """
    if num_frames is None:
        for v in rotations.values():
            num_frames = v.shape[0]
            break

    mixamo_rotations: dict[str, np.ndarray] = {}
    for internal_name, mixamo_name in INTERNAL_TO_MIXAMO.items():
        if internal_name in rotations:
            mixamo_rotations[mixamo_name] = rotations[internal_name]
        else:
            # Identity quaternion for missing bones
            quats = np.zeros((num_frames, 4))
            quats[:, 3] = 1.0
            mixamo_rotations[mixamo_name] = quats

    return {
        "rotations": mixamo_rotations,
        "bone_names": list(mixamo_rotations.keys()),
        "hierarchy": MIXAMO_HIERARCHY,
        "num_frames": num_frames,
    }
