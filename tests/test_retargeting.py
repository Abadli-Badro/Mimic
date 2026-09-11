"""Tests for retargeting modules (skeleton, retarget)."""

from __future__ import annotations

import numpy as np

from mimic.retargeting.retarget import (
    INTERNAL_TO_MIXAMO,
    load_mapping,
    retarget,
)
from mimic.retargeting.skeleton import Skeleton, create_mediapipe_skeleton


class TestSkeleton:
    def test_add_bone(self) -> None:
        skel = Skeleton()
        skel.add_bone("Hips")
        assert "Hips" in skel.bones
        assert skel.bones["Hips"].parent is None

    def test_add_bone_with_parent(self) -> None:
        skel = Skeleton()
        skel.add_bone("Hips")
        skel.add_bone("Spine", parent="Hips")
        assert skel.bones["Spine"].parent == "Hips"
        assert "Spine" in skel.bones["Hips"].children

    def test_get_parent(self) -> None:
        skel = Skeleton()
        skel.add_bone("Hips")
        skel.add_bone("Spine", parent="Hips")
        assert skel.get_parent("Spine") == "Hips"
        assert skel.get_parent("Hips") is None

    def test_get_parent_unknown_bone(self) -> None:
        skel = Skeleton()
        assert skel.get_parent("missing") is None

    def test_get_children(self) -> None:
        skel = Skeleton()
        skel.add_bone("Hips")
        skel.add_bone("Spine", parent="Hips")
        skel.add_bone("LeftUpLeg", parent="Hips")
        children = skel.get_children("Hips")
        assert set(children) == {"Spine", "LeftUpLeg"}

    def test_get_children_unknown_bone(self) -> None:
        skel = Skeleton()
        assert skel.get_children("missing") == []

    def test_create_mediapipe_skeleton(self) -> None:
        skel = create_mediapipe_skeleton()
        assert isinstance(skel, Skeleton)
        assert len(skel.bones) > 0


class TestRetarget:
    def test_load_mapping(self) -> None:
        mapping = load_mapping()
        assert isinstance(mapping, dict)
        assert "pelvis" in mapping
        assert mapping["pelvis"] == "mixamorig:Hips"

    def test_internal_to_mixamo_count(self) -> None:
        assert len(INTERNAL_TO_MIXAMO) > 0

    def test_retarget_remaps_bone_names(self) -> None:
        rotations = {
            "pelvis": np.zeros((5, 4)),
            "spine": np.zeros((5, 4)),
        }
        result = retarget(rotations, num_frames=5)
        assert "mixamorig:Hips" in result["rotations"]
        assert "mixamorig:Spine" in result["rotations"]

    def test_retarget_creates_identity_for_missing_bones(self) -> None:
        rotations = {"pelvis": np.zeros((3, 4))}
        result = retarget(rotations, num_frames=3)
        # Should have identity quaternions for bones not in input
        for bone_name in result["bone_names"]:
            quats = result["rotations"][bone_name]
            assert quats.shape == (3, 4)

    def test_retarget_preserves_rotations(self) -> None:
        q = np.zeros((2, 4))
        q[:, 3] = 1.0
        rotations = {"pelvis": q}
        result = retarget(rotations, num_frames=2)
        np.testing.assert_array_equal(result["rotations"]["mixamorig:Hips"], q)

    def test_retarget_output_structure(self) -> None:
        rotations = {"pelvis": np.zeros((5, 4))}
        result = retarget(rotations, num_frames=5)
        assert "rotations" in result
        assert "bone_names" in result
        assert "hierarchy" in result
        assert "num_frames" in result
        assert result["num_frames"] == 5
