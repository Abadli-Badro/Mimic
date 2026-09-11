"""Intermediate BVH output for debug checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_bvh(
    rotations: list[dict[str, np.ndarray]],
    fps: float,
    output_path: Path,
) -> None:
    """Write bone rotations to a BVH file.

    Args:
        rotations: Per-frame rotation dicts keyed by bone name.
        fps: Frame rate.
        output_path: Path to write the .bvh file.
    """
    # TODO: implement BVH writing
    raise NotImplementedError
