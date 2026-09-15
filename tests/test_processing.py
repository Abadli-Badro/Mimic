"""Tests for processing modules (smoothing, rotation_solver)."""

from __future__ import annotations

import numpy as np
import pytest

from mimic.processing.smoothing import OneEuroFilter, smooth_landmarks, smooth_landmarks_array


class TestOneEuroFilter:
    def test_filter_returns_same_shape(self) -> None:
        filt = OneEuroFilter(fps=30.0)
        for i in range(10):
            result = filt(1.0, timestamp=i / 30.0)
        assert isinstance(result, float)

    def test_filter_constant_signal_converges(self) -> None:
        filt = OneEuroFilter(fps=30.0, min_cutoff=1.0, beta=0.0)
        # Feed constant value — output should converge to input
        for i in range(100):
            result = filt(5.0, timestamp=i / 30.0)
        assert pytest.approx(result, abs=0.1) == 5.0

    def test_filter_responds_to_change(self) -> None:
        filt = OneEuroFilter(fps=30.0, min_cutoff=1.0, beta=0.01)
        # Establish steady state
        for i in range(50):
            filt(0.0, timestamp=i / 30.0)
        # Sudden change
        result = filt(10.0, timestamp=50 / 30.0)
        # Should not instantly jump to 10 but should move toward it
        assert 0.0 < result < 10.0

    def test_filter_low_cutoff_is_smoothing(self) -> None:
        low = OneEuroFilter(fps=30.0, min_cutoff=0.5, beta=0.0)
        high = OneEuroFilter(fps=30.0, min_cutoff=5.0, beta=0.0)
        for i in range(20):
            low(0.0, timestamp=i / 30.0)
            high(0.0, timestamp=i / 30.0)
        low_result = low(10.0, timestamp=20 / 30.0)
        high_result = high(10.0, timestamp=20 / 30.0)
        # Low cutoff should lag more (smaller change)
        assert low_result < high_result


class TestSmoothLandmarksArray:
    def test_smooth_preserves_shape(self) -> None:
        landmarks = np.random.rand(30, 33, 3).astype(np.float64)
        result = smooth_landmarks_array(landmarks, fps=30.0)
        assert result.shape == landmarks.shape

    def test_smooth_reduces_jitter(self) -> None:
        # Constant signal with noise
        base = np.ones((50, 33, 3)) * 5.0
        noise = np.random.RandomState(42).randn(50, 33, 3) * 0.5
        landmarks = base + noise
        result = smooth_landmarks_array(landmarks, fps=30.0, min_cutoff=0.5)
        # Smoothed signal should have lower variance than noisy input
        assert result[10:].std() < landmarks[10:].std()

    def test_smooth_constant_signal_stays_constant(self) -> None:
        landmarks = np.ones((20, 33, 3)) * 3.0
        result = smooth_landmarks_array(landmarks, fps=30.0)
        # After initial transient, should be close to 3.0
        np.testing.assert_allclose(result[10:], 3.0, atol=0.5)

    def test_smooth_single_frame(self) -> None:
        landmarks = np.random.rand(1, 33, 3).astype(np.float64)
        result = smooth_landmarks_array(landmarks, fps=30.0)
        assert result.shape == (1, 33, 3)


class TestSmoothLandmarksStub:
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

    def test_solve_rotations_unit_quaternions(self) -> None:
        from mimic.processing.rotation_solver import solve_rotations

        landmarks = np.random.rand(5, 33, 3).astype(np.float32)
        visibility = np.ones((5, 33), dtype=np.float32)
        result = solve_rotations(landmarks, visibility)

        for bone_name in result["bone_names"]:
            quats = result["rotations"][bone_name]
            norms = np.linalg.norm(quats, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_bone_hierarchy_covers_all_bones(self) -> None:
        from mimic.processing.rotation_solver import BONE_HIERARCHY, BONE_ORDER

        for bone in BONE_ORDER:
            assert bone in BONE_HIERARCHY
        # Root should have None parent
        assert BONE_HIERARCHY["pelvis"] is None
