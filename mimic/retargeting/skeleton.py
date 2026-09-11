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


def _down() -> np.ndarray:
    return np.array([0.0, -1.0, 0.0])


def _left() -> np.ndarray:
    return np.array([-1.0, 0.0, 0.0])


def _right() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])


def create_mediapipe_skeleton() -> Skeleton:
    """Create the internal skeleton matching MediaPipe pose landmarks.

    Y-up coordinate system (after MediaPipe Y-flip).
    Bones point from parent joint to child joint in rest/T-pose.
    """
    skel = Skeleton()

    # Spine chain (Y-up: positive Y = upward)
    skel.add_bone("pelvis", parent=None, rest_direction=_up())
    skel.add_bone("spine", parent="pelvis", rest_direction=_up())
    skel.add_bone("spine1", parent="spine", rest_direction=_up())
    skel.add_bone("chest", parent="spine1", rest_direction=_up())
    skel.add_bone("neck", parent="chest", rest_direction=_up())
    skel.add_bone("head", parent="neck", rest_direction=_up())

    # Shoulders
    skel.add_bone("left_shoulder", parent="chest", rest_direction=_left())
    skel.add_bone("right_shoulder", parent="chest", rest_direction=_right())

    # Left arm (T-pose: extends left, -X)
    skel.add_bone("left_upper_arm", parent="left_shoulder", rest_direction=_left())
    skel.add_bone("left_lower_arm", parent="left_upper_arm", rest_direction=_left())
    skel.add_bone("left_hand", parent="left_lower_arm", rest_direction=_left())

    # Right arm (T-pose: extends right, +X)
    skel.add_bone("right_upper_arm", parent="right_shoulder", rest_direction=_right())
    skel.add_bone("right_lower_arm", parent="right_upper_arm", rest_direction=_right())
    skel.add_bone("right_hand", parent="right_lower_arm", rest_direction=_right())

    # Hips
    skel.add_bone("left_hip", parent="pelvis", rest_direction=np.array([-0.3, -1.0, 0.0]))
    skel.add_bone("right_hip", parent="pelvis", rest_direction=np.array([0.3, -1.0, 0.0]))

    # Left leg (extends down, -Y; slight forward bend at knee/ankle)
    skel.add_bone(
        "left_upper_leg", parent="left_hip",
        rest_direction=np.array([0.0, -1.0, 0.0]),
    )
    skel.add_bone(
        "left_lower_leg", parent="left_upper_leg",
        rest_direction=np.array([0.0, -1.0, 0.05]),
    )
    skel.add_bone(
        "left_foot", parent="left_lower_leg",
        rest_direction=np.array([0.0, -1.0, 0.2]),
    )

    # Right leg (mirrors left)
    skel.add_bone(
        "right_upper_leg", parent="right_hip",
        rest_direction=np.array([0.0, -1.0, 0.0]),
    )
    skel.add_bone(
        "right_lower_leg", parent="right_upper_leg",
        rest_direction=np.array([0.0, -1.0, 0.05]),
    )
    skel.add_bone(
        "right_foot", parent="right_lower_leg",
        rest_direction=np.array([0.0, -1.0, 0.2]),
    )

    return skel
