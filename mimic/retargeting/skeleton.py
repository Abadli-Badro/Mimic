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


def _up() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0])


def _left() -> np.ndarray:
    return np.array([-1.0, 0.0, 0.0])


def _right() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


def _forward() -> np.ndarray:
    return np.array([0.0, 0.0, -1.0])


def create_mediapipe_skeleton() -> Skeleton:
    """Create the internal skeleton matching MediaPipe pose landmarks.

    Bones are defined as (parent_joint, child_joint) pairs with rest-pose
    direction vectors pointing from parent to child.
    """
    skel = Skeleton()

    # Spine chain
    skel.add_bone("pelvis", parent=None, rest_direction=_up())
    skel.add_bone("spine", parent="pelvis", rest_direction=_up())
    skel.add_bone("chest", parent="spine", rest_direction=_up())
    skel.add_bone("neck", parent="chest", rest_direction=_up())
    skel.add_bone("head", parent="neck", rest_direction=_up())

    # Left arm
    skel.add_bone("left_shoulder", parent="chest", rest_direction=_left())
    skel.add_bone("left_upper_arm", parent="left_shoulder", rest_direction=_left())
    skel.add_bone("left_lower_arm", parent="left_upper_arm", rest_direction=_left())
    skel.add_bone("left_hand", parent="left_lower_arm", rest_direction=_left())

    # Right arm
    skel.add_bone("right_shoulder", parent="chest", rest_direction=_right())
    skel.add_bone("right_upper_arm", parent="right_shoulder", rest_direction=_right())
    skel.add_bone("right_lower_arm", parent="right_upper_arm", rest_direction=_right())
    skel.add_bone("right_hand", parent="right_lower_arm", rest_direction=_right())

    # Left leg
    skel.add_bone("left_hip", parent="pelvis", rest_direction=_left())
    skel.add_bone(
        "left_upper_leg", parent="left_hip",
        rest_direction=np.array([-0.2, -1.0, 0.0]),
    )
    skel.add_bone(
        "left_lower_leg", parent="left_upper_leg",
        rest_direction=np.array([0.0, -1.0, 0.1]),
    )
    skel.add_bone(
        "left_foot", parent="left_lower_leg",
        rest_direction=np.array([0.0, -1.0, 0.2]),
    )

    # Right leg
    skel.add_bone("right_hip", parent="pelvis", rest_direction=_right())
    skel.add_bone(
        "right_upper_leg", parent="right_hip",
        rest_direction=np.array([0.2, -1.0, 0.0]),
    )
    skel.add_bone(
        "right_lower_leg", parent="right_upper_leg",
        rest_direction=np.array([0.0, -1.0, 0.1]),
    )
    skel.add_bone(
        "right_foot", parent="right_lower_leg",
        rest_direction=np.array([0.0, -1.0, 0.2]),
    )

    return skel
