"""Internal skeleton hierarchy definition."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bone:
    name: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class Skeleton:
    bones: dict[str, Bone] = field(default_factory=dict)

    def add_bone(self, name: str, parent: str | None = None) -> None:
        bone = Bone(name=name, parent=parent)
        self.bones[name] = bone
        if parent and parent in self.bones:
            self.bones[parent].children.append(name)

    def get_parent(self, bone_name: str) -> str | None:
        return self.bones[bone_name].parent if bone_name in self.bones else None

    def get_children(self, bone_name: str) -> list[str]:
        return self.bones[bone_name].children if bone_name in self.bones else []


def create_mediapipe_skeleton() -> Skeleton:
    """Create the internal skeleton matching MediaPipe pose landmarks."""
    # TODO: define full MediaPipe landmark hierarchy
    skel = Skeleton()
    return skel
