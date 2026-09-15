"""Convert landmark joint positions into bone rotation quaternions.

Uses measured torso/head frames and parent-local swing rotations for limbs.
Single segment directions cannot recover anatomical limb twist.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from mimic.validation import hierarchy as validate_hierarchy, number
from mimic.errors import MimicError

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
    if bone_name in {"pelvis", "spine", "spine1", "chest"}:
        return _dir(_mid(lm[23], lm[24]), _mid(lm[11], lm[12]))
    if bone_name in {"neck", "head"}:
        return _dir(_mid(lm[11], lm[12]), _mid(lm[7], lm[8]))
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
    nr, nc = np.linalg.norm(rest_dir), np.linalg.norm(current_dir)
    if min(nr, nc) < 1e-8:
        return Rotation.identity()
    rest, curr = rest_dir / nr, current_dir / nc
    dot = np.clip(np.dot(rest, curr), -1.0, 1.0)
    axis = np.cross(rest, curr)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        if dot > 0:
            return Rotation.identity()
        axis = np.cross(rest, np.eye(3)[np.argmin(np.abs(rest))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis))
    return Rotation.from_rotvec(np.arctan2(norm, dot) * axis / norm)


BONE_LANDMARKS = {
    **{b: [23, 24, 11, 12] for b in ["pelvis", "spine", "spine1", "chest"]},
    "neck": [11, 12, 7, 8], "head": [7, 8, 0],
    "left_shoulder": [11, 12], "right_shoulder": [11, 12],
    "left_upper_arm": [11, 13], "left_lower_arm": [13, 15],
    "left_hand": [15, 19], "right_upper_arm": [12, 14],
    "right_lower_arm": [14, 16], "right_hand": [16, 20],
    "left_hip": [23, 25], "left_upper_leg": [25, 27],
    "left_lower_leg": [27, 31], "left_foot": [29, 31],
    "right_hip": [24, 26], "right_upper_leg": [26, 28],
    "right_lower_leg": [28, 32], "right_foot": [30, 32],
}


def _body_frame(x, y):
    """Orthonormal frame from lateral and upward vectors; None if degenerate."""
    if np.linalg.norm(y) < 1e-8:
        return None
    y = y / np.linalg.norm(y)
    x = x - y * np.dot(x, y)
    if np.linalg.norm(x) < 1e-8:
        return None
    x /= np.linalg.norm(x)
    return Rotation.from_matrix(np.column_stack((x, y, np.cross(x, y))))


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

    validate_hierarchy(BONE_HIERARCHY)
    number(visibility_threshold, 'visibility threshold', maximum=1, inclusive=True)
    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 3) or not len(landmarks):
        raise ValueError("Expected nonempty (frames, 33, 3) landmarks")
    if visibility is not None and np.shape(visibility) != landmarks.shape[:2]:
        raise ValueError("Visibility must have shape (frames, 33)")
    if visibility is not None:
        visibility = np.asarray(visibility, dtype=float)
        if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
            raise MimicError('invalid_visibility', 'Visibility must contain finite values between 0 and 1.')
    trusted = np.ones(landmarks.shape[:2], dtype=bool) if visibility is None else visibility >= visibility_threshold
    if not np.isfinite(landmarks[trusted]).all():
        raise MimicError('invalid_landmarks', 'Trusted landmarks contain nonfinite coordinates.')
    skel = create_mediapipe_skeleton()
    num_frames = len(landmarks)
    bone_rotations = {name: np.tile([0., 0., 0., 1.], (num_frames, 1)) for name in BONE_ORDER}

    for frame_idx, lm in enumerate(landmarks):
        world_rots = {}
        for bone_name in BONE_ORDER:
            parent = BONE_HIERARCHY[bone_name]
            parent_world = world_rots[parent] if parent else Rotation.identity()
            indices = BONE_LANDMARKS[bone_name]
            valid = np.isfinite(lm[indices]).all()
            if visibility is not None:
                valid = valid and bool(np.all(visibility[frame_idx, indices] >= visibility_threshold))
            local_rot = (Rotation.from_quat(bone_rotations[bone_name][frame_idx - 1])
                         if frame_idx else Rotation.identity())
            if valid:
                direction = _compute_bone_direction(bone_name, lm)
                measured = None
                if bone_name in {"pelvis", "spine", "spine1", "chest"}:
                    lateral = lm[23] - lm[24] if bone_name == "pelvis" else lm[11] - lm[12]
                    measured = _body_frame(lateral, direction)
                elif bone_name == "head":
                    x = lm[7] - lm[8]
                    forward = lm[0] - _mid(lm[7], lm[8])
                    measured = _body_frame(x, np.cross(forward, x))
                if measured is not None:
                    local_rot = parent_world.inv() * measured
                elif np.linalg.norm(direction) > 1e-8 and bone_name != "head":
                    local_rot = _swing_rotation(
                        skel.bones[bone_name].rest_direction,
                        parent_world.inv().apply(direction),
                    )
            quat = local_rot.as_quat()
            if frame_idx and np.dot(quat, bone_rotations[bone_name][frame_idx - 1]) < 0:
                quat = -quat
            bone_rotations[bone_name][frame_idx] = quat
            world_rots[bone_name] = parent_world * local_rot

    return {
        "rotations": bone_rotations,
        "bone_names": list(BONE_ORDER),
        "hierarchy": BONE_HIERARCHY,
        "num_frames": num_frames,
    }
