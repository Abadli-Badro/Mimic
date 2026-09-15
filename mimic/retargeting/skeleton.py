"""Internal skeleton hierarchy definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bone:
    name: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    rest_direction: np.ndarray | None = None  # unit vector in rest pose


@dataclass
class Skeleton:
    bones: dict[str, Bone] = field(default_factory=dict)

    def add_bone(
        self,
        name: str,
        parent: str | None = None,
        rest_direction: np.ndarray | None = None,
    ) -> None:
        bone = Bone(name=name, parent=parent, rest_direction=rest_direction)
        self.bones[name] = bone
        if parent and parent in self.bones:
            self.bones[parent].children.append(name)

    def get_parent(self, bone_name: str) -> str | None:
        return self.bones[bone_name].parent if bone_name in self.bones else None

    def get_children(self, bone_name: str) -> list[str]:
        return self.bones[bone_name].children if bone_name in self.bones else []


# Canonical Y-up, +Z-forward T-pose, in meters. Anatomical left is +X.
# Offsets locate joints; rest directions describe the segment leaving a joint.
JOINT_OFFSETS = {
    "pelvis": (0, 0.95, 0), "spine": (0, .12, 0),
    "spine1": (0, .12, 0), "chest": (0, .12, 0),
    "neck": (0, .14, 0), "head": (0, .10, 0),
    "left_shoulder": (.08, .10, 0), "left_upper_arm": (.12, 0, 0),
    "left_lower_arm": (.28, 0, 0), "left_hand": (.25, 0, 0),
    "right_shoulder": (-.08, .10, 0), "right_upper_arm": (-.12, 0, 0),
    "right_lower_arm": (-.28, 0, 0), "right_hand": (-.25, 0, 0),
    "left_hip": (.10, 0, 0), "left_upper_leg": (0, -.43, 0),
    "left_lower_leg": (0, -.42, 0), "left_foot": (0, 0, .15),
    "right_hip": (-.10, 0, 0), "right_upper_leg": (0, -.43, 0),
    "right_lower_leg": (0, -.42, 0), "right_foot": (0, 0, .15),
}


def create_mediapipe_skeleton() -> Skeleton:
    """Canonical skeleton with identity local rest rotations.

    Legacy leg names are retained: hip=thigh, upper_leg=shin,
    lower_leg=foot, foot=toes (matching the Mixamo mapping).
    """
    from mimic.processing.rotation_solver import BONE_HIERARCHY, BONE_ORDER

    skel = Skeleton()
    for name in BONE_ORDER:
        if any(part in name for part in ("shoulder", "arm", "hand")):
            direction = (1, 0, 0) if name.startswith("left") else (-1, 0, 0)
        elif name.endswith(("lower_leg", "foot")):
            direction = (0, 0, 1)
        elif name.endswith(("hip", "upper_leg")):
            direction = (0, -1, 0)
        else:
            direction = (0, 1, 0)
        skel.add_bone(name, BONE_HIERARCHY[name], np.array(direction, dtype=float))
    return skel
