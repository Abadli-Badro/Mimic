"""Convert landmark joint positions into bone rotation quaternions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Mapping from MediaPipe landmark indices to skeleton joint names
# (parent_joint_idx, child_joint_idx, bone_name)
BONE_DEFINITIONS = [
    # Spine
    (23, 11, "pelvis"),       # left_hip -> left_shoulder (approximates pelvis)
    (11, 0, "spine"),         # left_shoulder -> nose (approximates spine)
    (0, 0, "chest"),          # nose -> nose (head reference)
    (0, 0, "neck"),           # placeholder
    (0, 0, "head"),           # placeholder
    # Left arm
    (11, 13, "left_upper_arm"),
    (13, 15, "left_lower_arm"),
    (15, 19, "left_hand"),    # wrist -> index
    # Right arm
    (12, 14, "right_upper_arm"),
    (14, 16, "right_lower_arm"),
    (16, 20, "right_hand"),   # wrist -> index
    # Left leg
    (23, 25, "left_upper_leg"),
    (25, 27, "left_lower_leg"),
    (27, 31, "left_foot"),    # ankle -> foot_index
    # Right leg
    (24, 26, "right_upper_leg"),
    (26, 28, "right_lower_leg"),
    (28, 32, "right_foot"),   # ankle -> foot_index
]

# Simplified skeleton hierarchy
BONE_HIERARCHY = {
    "pelvis": None,
    "spine": "pelvis",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_upper_arm": "chest",
    "left_lower_arm": "left_upper_arm",
    "left_hand": "left_lower_arm",
    "right_upper_arm": "chest",
    "right_lower_arm": "right_upper_arm",
    "right_hand": "right_lower_arm",
    "left_upper_leg": "pelvis",
    "left_lower_leg": "left_upper_leg",
    "left_foot": "left_lower_leg",
    "right_upper_leg": "pelvis",
    "right_lower_leg": "right_upper_leg",
    "right_foot": "right_lower_leg",
}


def _compute_bone_direction(
    landmarks: np.ndarray,
    parent_idx: int,
    child_idx: int,
) -> np.ndarray:
    """Compute a unit direction vector from parent joint to child joint.

    Args:
        landmarks: Array of shape (33, 3) — world landmarks for one frame.
        parent_idx: MediaPipe landmark index for the parent joint.
        child_idx: MediaPipe landmark index for the child joint.

    Returns:
        Unit direction vector (3,).
    """
    direction = landmarks[child_idx] - landmarks[parent_idx]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    return direction / norm


def _solve_bone_rotation(
    current_direction: np.ndarray,
    rest_direction: np.ndarray,
) -> np.ndarray:
    """Compute the quaternion that rotates rest_direction to current_direction.

    Returns quaternion in (x, y, z, w) format.
    """
    current = current_direction.reshape(1, 3)
    rest = rest_direction.reshape(1, 3)

    try:
        r, _ = Rotation.align_vectors(rest, current)
        return r.as_quat()  # (x, y, z, w)
    except Exception:
        return np.array([0.0, 0.0, 0.0, 1.0])


def solve_rotations(
    landmarks: np.ndarray,
    visibility: np.ndarray | None = None,
    visibility_threshold: float = 0.5,
) -> dict:
    """Compute bone rotations from per-frame world landmark positions.

    Args:
        landmarks: Array of shape (num_frames, 33, 3).
        visibility: Optional array of shape (num_frames, 33).
        visibility_threshold: Min visibility to trust a landmark.

    Returns:
        Dict with keys:
          - "rotations": dict mapping bone_name -> array of shape (num_frames, 4) quaternions
          - "bone_names": list of bone names
          - "hierarchy": dict mapping bone -> parent bone
          - "num_frames": int
    """
    from mimic.retargeting.skeleton import create_mediapipe_skeleton

    skel = create_mediapipe_skeleton()
    num_frames = landmarks.shape[0]

    # Initialize rotation storage
    bone_rotations: dict[str, np.ndarray] = {
        bone_name: np.zeros((num_frames, 4)) for bone_name in BONE_HIERARCHY
    }
    for bone_name in bone_rotations:
        bone_rotations[bone_name][:, 3] = 1.0  # identity quaternion

    for frame_idx in range(num_frames):
        lm = landmarks[frame_idx]
        vis = visibility[frame_idx] if visibility is not None else np.ones(33)

        for parent_idx, child_idx, bone_name in BONE_DEFINITIONS:
            # Check if both joints are visible
            if parent_idx != child_idx:
                if vis[parent_idx] < visibility_threshold or vis[child_idx] < visibility_threshold:
                    continue

            direction = _compute_bone_direction(lm, parent_idx, child_idx)
            if np.allclose(direction, 0):
                continue

            # Get rest direction from skeleton
            bone = skel.bones.get(bone_name)
            if bone and bone.rest_direction is not None:
                rest_dir = bone.rest_direction / np.linalg.norm(bone.rest_direction)
            else:
                rest_dir = np.array([0.0, 1.0, 0.0])

            quat = _solve_bone_rotation(direction, rest_dir)
            bone_rotations[bone_name][frame_idx] = quat

    return {
        "rotations": bone_rotations,
        "bone_names": list(BONE_HIERARCHY.keys()),
        "hierarchy": BONE_HIERARCHY,
        "num_frames": num_frames,
    }
