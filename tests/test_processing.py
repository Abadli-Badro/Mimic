"""Tests for processing modules (smoothing, rotation_solver)."""

from __future__ import annotations

import numpy as np
import pytest

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
    def test_solve_rotations_not_implemented(self) -> None:
        from mimic.processing.rotation_solver import solve_rotations

        with pytest.raises(NotImplementedError):
            solve_rotations([{"nose": np.zeros(3)}])
