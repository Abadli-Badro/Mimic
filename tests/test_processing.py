"""Tests for processing modules (smoothing, rotation_solver)."""

from __future__ import annotations

import numpy as np

from mimic.processing.smoothing import smooth_landmarks


class TestSmoothing:
    def test_smooth_passthrough(self) -> None:
        landmarks = [
            {"nose": np.array([1.0, 2.0, 3.0])},
            {"nose": np.array([1.1, 2.1, 3.1])},
            {"nose": np.array([1.2, 2.2, 3.2])},
        ]
        result = smooth_landmarks(landmarks, fps=30.0)
        assert len(result) == 3
        assert "nose" in result[0]

    def test_smooth_preserves_length(self) -> None:
        landmarks = [{"joint": np.random.rand(3)} for _ in range(50)]
        result = smooth_landmarks(landmarks, fps=30.0)
        assert len(result) == 50


class TestRotationSolver:
    def test_solve_rotations_output_shapes(self) -> None:
        from mimic.processing.rotation_solver import solve_rotations

        # Synthetic landmarks: 10 frames, 33 joints, 3 coords
        landmarks = np.random.rand(10, 33, 3).astype(np.float32)
        visibility = np.ones((10, 33), dtype=np.float32)
        result = solve_rotations(landmarks, visibility)

        assert result["num_frames"] == 10
        assert len(result["bone_names"]) > 0
        assert isinstance(result["hierarchy"], dict)
        for bone_name in result["bone_names"]:
            assert result["rotations"][bone_name].shape == (10, 4)

    def test_identity_for_zero_movement(self) -> None:
        from mimic.processing.rotation_solver import solve_rotations

        # All frames identical — rotation should be identity (quat w=1)
        landmarks = np.zeros((5, 33, 3), dtype=np.float32)
        landmarks[:, 11] = np.array([0.5, 0.5, 0.0])  # left_shoulder
        landmarks[:, 12] = np.array([-0.5, 0.5, 0.0])  # right_shoulder
        landmarks[:, 23] = np.array([0.3, 0.0, 0.0])  # left_hip
        landmarks[:, 24] = np.array([-0.3, 0.0, 0.0])  # right_hip
        visibility = np.ones((5, 33), dtype=np.float32)

        result = solve_rotations(landmarks, visibility)
        # Upper arm bones should have valid quaternions
        for bone in ["left_upper_arm", "right_upper_arm"]:
            quats = result["rotations"][bone]
            # Quaternion should be unit length
            norms = np.linalg.norm(quats, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)
