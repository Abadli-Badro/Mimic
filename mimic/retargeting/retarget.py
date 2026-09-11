"""Retarget computed rotations onto a target Mixamo skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAPPING_PATH = Path(__file__).parent / "mixamo_mapping.json"


def load_mapping() -> dict[str, str]:
    """Load the internal-bone-to-Mixamo-bone name mapping."""
    return json.loads(MAPPING_PATH.read_text())


def retarget(
    rotations: list[dict[str, np.ndarray]],
    target_skeleton: dict | None = None,
) -> list[dict[str, np.ndarray]]:
    """Map internal bone rotations onto the target skeleton's bone names.

    Args:
        rotations: Per-frame rotation dicts from the rotation solver.
        target_skeleton: Optional target skeleton definition.

    Returns:
        Per-frame rotation dicts keyed by target skeleton bone names.
    """
    mapping = load_mapping()
    # TODO: apply rest-pose differences, coordinate system conversion
    result: list[dict[str, np.ndarray]] = []
    for frame_rot in rotations:
        remapped: dict[str, np.ndarray] = {}
        for internal_name, target_name in mapping.items():
            if internal_name in frame_rot:
                remapped[target_name] = frame_rot[internal_name]
        result.append(remapped)
    return result
