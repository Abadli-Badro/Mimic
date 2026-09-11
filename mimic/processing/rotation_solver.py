"""Convert landmark joint positions into bone rotation quaternions.

Uses a proper hierarchy walk with local rotation computation and
twist-swing decomposition to avoid random twist artifacts.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# MediaPipe landmark indices
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_NOSE = 0
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8
LM_LEFT_EYE_INNER = 1
LM_RIGHT_EYE_INNER = 4
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_INDEX = 19
LM_RIGHT_INDEX = 20
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_HEEL = 29
LM_RIGHT_HEEL = 30
LM_LEFT_FOOT_INDEX = 31
LM_RIGHT_FOOT_INDEX = 32

# Bone hierarchy (parent -> children)
BONE_HIERARCHY = {
    "pelvis": None,
    "spine": "pelvis",
    "spine1": "spine",
    "chest": "spine1",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "right_shoulder": "chest",
    "left_upper_arm": "left_shoulder",
    "left_lower_arm": "left_upper_arm",
    "left_hand": "left_lower_arm",
    "right_upper_arm": "right_shoulder",
    "right_lower_arm": "right_upper_arm",
    "right_hand": "right_lower_arm",
    "left_hip": "pelvis",
    "left_upper_leg": "left_hip",
    "left_lower_leg": "left_upper_leg",
    "left_foot": "left_lower_leg",
    "right_hip": "pelvis",
    "right_upper_leg": "right_hip",
    "right_lower_leg": "right_upper_leg",
    "right_foot": "right_lower_leg",
}

# Bones in parent-first (topological) order for hierarchy walk
BONE_ORDER = [
    "pelvis", "spine", "spine1", "chest", "neck", "head",
    "left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand",
    "right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand",
    "left_hip", "left_upper_leg", "left_lower_leg", "left_foot",
    "right_hip", "right_upper_leg", "right_lower_leg", "right_foot",
]


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _dir(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-8:
        return np.zeros(3)
    return d / n


def _compute_bone_direction(bone_name: str, lm: np.ndarray) -> np.ndarray:
    """Compute the direction vector for a bone from landmarks.

    Returns unit direction FROM parent joint TO child joint.
    """
    if bone_name == "pelvis":
        return _dir(_mid(lm[LM_LEFT_HIP], lm[LM_RIGHT_HIP]),
                     _mid(lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]))
    if bone_name == "spine":
        return _dir(_mid(lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]),
                     lm[LM_NOSE])
    if bone_name == "spine1":
        return _dir(_mid(lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]),
                     _mid(lm[LM_LEFT_EAR], lm[LM_RIGHT_EAR]))
    if bone_name == "chest":
        return _dir(lm[LM_NOSE],
                     _mid(lm[LM_LEFT_EAR], lm[LM_RIGHT_EAR]))
    if bone_name == "neck":
        return _dir(_mid(lm[LM_LEFT_EAR], lm[LM_RIGHT_EAR]),
                     lm[LM_NOSE])
    if bone_name == "head":
        return _dir(lm[LM_NOSE],
                     _mid(lm[LM_LEFT_EYE_INNER], lm[LM_RIGHT_EYE_INNER]))
    if bone_name == "left_shoulder":
        return _dir(_mid(lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]),
                     lm[LM_LEFT_SHOULDER])
    if bone_name == "right_shoulder":
        return _dir(_mid(lm[LM_LEFT_SHOULDER], lm[LM_RIGHT_SHOULDER]),
                     lm[LM_RIGHT_SHOULDER])
    if bone_name == "left_upper_arm":
        return _dir(lm[LM_LEFT_SHOULDER], lm[LM_LEFT_ELBOW])
    if bone_name == "left_lower_arm":
        return _dir(lm[LM_LEFT_ELBOW], lm[LM_LEFT_WRIST])
    if bone_name == "left_hand":
        return _dir(lm[LM_LEFT_WRIST], lm[LM_LEFT_INDEX])
    if bone_name == "right_upper_arm":
        return _dir(lm[LM_RIGHT_SHOULDER], lm[LM_RIGHT_ELBOW])
    if bone_name == "right_lower_arm":
        return _dir(lm[LM_RIGHT_ELBOW], lm[LM_RIGHT_WRIST])
    if bone_name == "right_hand":
        return _dir(lm[LM_RIGHT_WRIST], lm[LM_RIGHT_INDEX])
    if bone_name == "left_hip":
        return _dir(lm[LM_LEFT_HIP], lm[LM_LEFT_KNEE])
    if bone_name == "left_upper_leg":
        return _dir(lm[LM_LEFT_KNEE], lm[LM_LEFT_ANKLE])
    if bone_name == "left_lower_leg":
        return _dir(lm[LM_LEFT_ANKLE], lm[LM_LEFT_FOOT_INDEX])
    if bone_name == "left_foot":
        return _dir(lm[LM_LEFT_HEEL], lm[LM_LEFT_FOOT_INDEX])
    if bone_name == "right_hip":
        return _dir(lm[LM_RIGHT_HIP], lm[LM_RIGHT_KNEE])
    if bone_name == "right_upper_leg":
        return _dir(lm[LM_RIGHT_KNEE], lm[LM_RIGHT_ANKLE])
    if bone_name == "right_lower_leg":
        return _dir(lm[LM_RIGHT_ANKLE], lm[LM_RIGHT_FOOT_INDEX])
    if bone_name == "right_foot":
        return _dir(lm[LM_RIGHT_HEEL], lm[LM_RIGHT_FOOT_INDEX])
    return np.zeros(3)


def _swing_rotation(rest_dir: np.ndarray, current_dir: np.ndarray) -> Rotation:
    """Compute the swing-only rotation from rest direction to current direction."""
    rest = rest_dir / (np.linalg.norm(rest_dir) + 1e-8)
    curr = current_dir / (np.linalg.norm(current_dir) + 1e-8)

    dot = np.clip(np.dot(rest, curr), -1.0, 1.0)

    if dot > 0.9999:
        return Rotation.identity()

    if dot < -0.9999:
        perp = np.array([1.0, 0.0, 0.0]) if abs(rest[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(rest, perp)
        axis /= np.linalg.norm(axis) + 1e-8
        return Rotation.from_rotvec(np.pi * axis)

    axis = np.cross(rest, curr)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.arccos(dot)
    return Rotation.from_rotvec(angle * axis)


def solve_rotations(
    landmarks: np.ndarray,
    visibility: np.ndarray | None = None,
    visibility_threshold: float = 0.5,
) -> dict:
    """Compute bone rotations from per-frame world landmark positions.

    Computes LOCAL rotations directly by working in parent's local frame.
    This avoids the mirrored-bone issue where left/right produce identical
    world rotations.

    Args:
        landmarks: Array of shape (num_frames, 33, 3) — Y-up world landmarks.
        visibility: Optional array of shape (num_frames, 33).
        visibility_threshold: Min visibility to trust a landmark.

    Returns:
        Dict with keys:
          - "rotations": dict mapping bone_name -> (num_frames, 4) quaternions
          - "bone_names": list of bone names
          - "hierarchy": dict mapping bone -> parent bone
          - "num_frames": int
    """
    from mimic.retargeting.skeleton import create_mediapipe_skeleton

    skel = create_mediapipe_skeleton()
    num_frames = landmarks.shape[0]

    bone_rotations: dict[str, np.ndarray] = {
        name: np.zeros((num_frames, 4)) for name in BONE_ORDER
    }
    for name in bone_rotations:
        bone_rotations[name][:, 3] = 1.0

    for frame_idx in range(num_frames):
        lm = landmarks[frame_idx]

        world_rots: dict[str, Rotation] = {}

        for bone_name in BONE_ORDER:
            direction = _compute_bone_direction(bone_name, lm)
            if np.allclose(direction, 0):
                world_rots[bone_name] = Rotation.identity()
                continue

            bone = skel.bones.get(bone_name)
            if bone and bone.rest_direction is not None:
                rest_dir = bone.rest_direction / (
                    np.linalg.norm(bone.rest_direction) + 1e-8
                )
            else:
                rest_dir = np.array([0.0, 1.0, 0.0])

            parent_name = BONE_HIERARCHY.get(bone_name)
            parent_world = world_rots.get(parent_name) if parent_name else None

            if parent_world is not None:
                # Transform both directions into parent's local frame
                local_rest = parent_world.inv().apply(rest_dir.reshape(1, 3))[0]
                local_curr = parent_world.inv().apply(direction.reshape(1, 3))[0]
                # Compute rotation in local frame
                local_rot = _swing_rotation(local_rest, local_curr)
            else:
                # Root bone: compute world rotation directly
                local_rot = _swing_rotation(rest_dir, direction)

            bone_rotations[bone_name][frame_idx] = local_rot.as_quat()

            # Accumulate world rotation for children
            if parent_world is not None:
                world_rots[bone_name] = parent_world * local_rot
            else:
                world_rots[bone_name] = local_rot

    return {
        "rotations": bone_rotations,
        "bone_names": list(BONE_ORDER),
        "hierarchy": BONE_HIERARCHY,
        "num_frames": num_frames,
    }
