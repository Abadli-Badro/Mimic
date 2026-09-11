"""Convert landmark joint positions into bone rotation quaternions."""

from __future__ import annotations

import numpy as np


def solve_rotations(
    landmarks: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Compute bone rotations from per-frame joint positions.

    Args:
        landmarks: Per-frame landmark dictionaries mapping name to (x, y, z).

    Returns:
        List of rotation dicts, one per frame. Each dict maps bone name
        to a quaternion as numpy array (x, y, z, w).
    """
    # TODO: implement rotation solving via direction vectors + quaternions
    raise NotImplementedError
