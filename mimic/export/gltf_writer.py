"""Primary glTF/GLB export via pygltflib (pure Python, no external engine)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_gltf(
    rotations: list[dict[str, np.ndarray]],
    fps: float,
    output_path: Path,
    glb: bool = True,
) -> None:
    """Write bone rotations to a glTF or GLB file.

    Args:
        rotations: Per-frame rotation dicts keyed by Mixamo bone names.
        fps: Frame rate.
        output_path: Path to write the .gltf or .glb file.
        glb: If True, write binary GLB format. If False, write .gltf.
    """
    # TODO: implement glTF animation writing via pygltflib
    raise NotImplementedError
