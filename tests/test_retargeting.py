"""Tests for retargeting modules (skeleton, retarget)."""

from __future__ import annotations

import numpy as np

from mimic.retargeting.retarget import load_mapping, retarget
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


class TestRetarget:
    def test_load_mapping(self) -> None:
        mapping = load_mapping()
        assert isinstance(mapping, dict)
        assert "Hips" in mapping
        assert mapping["Hips"] == "mixamorig:Hips"

    def test_load_mapping_count(self) -> None:
        mapping = load_mapping()
        assert len(mapping) > 0

    def test_retarget_remaps_bone_names(self) -> None:
        rotations = [
            {
                "Hips": np.array([0.0, 0.0, 0.0, 1.0]),
                "Spine": np.array([0.0, 0.0, 0.0, 1.0]),
                "UnknownBone": np.array([0.0, 0.0, 0.0, 1.0]),
            }
        ]
        result = retarget(rotations)
        assert len(result) == 1
        assert "mixamorig:Hips" in result[0]
        assert "mixamorig:Spine" in result[0]
        assert "UnknownBone" not in result[0]

    def test_retarget_empty_input(self) -> None:
        result = retarget([])
        assert result == []

    def test_retarget_preserves_rotations(self) -> None:
        q = np.array([0.1, 0.2, 0.3, 0.9])
        rotations = [{"Hips": q}]
        result = retarget(rotations)
        np.testing.assert_array_equal(result[0]["mixamorig:Hips"], q)
