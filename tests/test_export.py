"""Tests for export modules (bvh_writer, gltf_writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.export.bvh_writer import _get_joint_order, write_bvh
from mimic.export.gltf_writer import _collect_bone_order, write_gltf


def _make_fake_rotations(num_frames: int = 10) -> dict[str, np.ndarray]:
    """Create fake rotation data for testing."""
    bones = [
        "pelvis", "spine", "chest", "neck", "head",
        "left_upper_arm", "left_lower_arm", "left_hand",
        "right_upper_arm", "right_lower_arm", "right_hand",
        "left_upper_leg", "left_lower_leg", "left_foot",
        "right_upper_leg", "right_lower_leg", "right_foot",
    ]
    rotations = {}
    for bone in bones:
        quats = np.zeros((num_frames, 4))
        quats[:, 3] = 1.0  # identity quaternion
        rotations[bone] = quats
    return rotations


class TestBVHWriter:
    def test_write_bvh_creates_file(self, tmp_path: Path) -> None:
        rotations = _make_fake_rotations(5)
        out = tmp_path / "test.bvh"
        write_bvh(rotations, fps=30.0, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_bvh_content(self, tmp_path: Path) -> None:
        rotations = _make_fake_rotations(3)
        out = tmp_path / "test.bvh"
        write_bvh(rotations, fps=30.0, output_path=out)
        content = out.read_text()
        assert "HIERARCHY" in content
        assert "MOTION" in content
        assert "Frames: 3" in content
        assert "ROOT pelvis" in content

    def test_joint_order(self) -> None:
        order = _get_joint_order()
        assert order[0] == "pelvis"
        assert len(order) > 0


class TestGLTFWriter:
    def test_write_glb_creates_file(self, tmp_path: Path) -> None:
        rotations = _make_fake_rotations(5)
        # Remap to Mixamo names
        from mimic.retargeting.retarget import INTERNAL_TO_MIXAMO
        mixamo_rot = {}
        for internal, mixamo in INTERNAL_TO_MIXAMO.items():
            if internal in rotations:
                mixamo_rot[mixamo] = rotations[internal]

        out = tmp_path / "test.glb"
        write_gltf(mixamo_rot, fps=30.0, output_path=out, glb=True)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_gltf_creates_file(self, tmp_path: Path) -> None:
        from mimic.retargeting.retarget import INTERNAL_TO_MIXAMO

        rotations = _make_fake_rotations(5)
        mixamo_rot = {}
        for internal, mixamo in INTERNAL_TO_MIXAMO.items():
            if internal in rotations:
                mixamo_rot[mixamo] = rotations[internal]

        out = tmp_path / "test.gltf"
        write_gltf(mixamo_rot, fps=30.0, output_path=out, glb=False)
        assert out.exists()

    def test_collect_bone_order(self) -> None:
        from mimic.retargeting.retarget import MIXAMO_HIERARCHY

        order = _collect_bone_order(MIXAMO_HIERARCHY)
        assert order[0] == "mixamorig:Hips"
        assert len(order) > 0
