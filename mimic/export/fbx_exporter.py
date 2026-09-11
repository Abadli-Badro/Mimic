"""Optional FBX export via bpy (Blender as in-process Python module).

This module is isolated so the rest of the codebase never depends on Blender.
Only use if FBX output is specifically required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_fbx(
    rotations: list[dict[str, np.ndarray]],
    fps: float,
    output_path: Path,
) -> None:
    """Export bone rotations to FBX via Blender's bpy module.

    Requires `pip install bpy` to be installed.

    Args:
        rotations: Per-frame rotation dicts keyed by Mixamo bone names.
        fps: Frame rate.
        output_path: Path to write the .fbx file.
    """
    # TODO: implement FBX export via bpy
    raise NotImplementedError(
        "FBX export requires bpy (Blender as Python module). "
        "Install with: pip install bpy"
    )
