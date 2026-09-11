"""Pipeline orchestrator — the single entrypoint for CLI and future API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.export.gltf_writer import write_gltf
from mimic.extraction.phase1 import extract as do_extract
from mimic.processing.phase2 import smooth as do_smooth
from mimic.processing.phase3 import solve as do_solve
from mimic.retargeting.retarget import retarget


def run(
    video_path: Path,
    output_path: Path | None = None,
    fps: int | None = None,
    format: str = "glb",
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> Path:
    """Run the full video-to-3D-animation pipeline.

    Args:
        video_path: Path to the input MP4 video.
        output_path: Path for the output file. Defaults to video name with .glb extension.
        fps: Target frame rate. None keeps source fps.
        format: Output format — "glb" (default) or "gltf".
        min_cutoff: One-Euro filter min cutoff.
        beta: One-Euro filter speed coefficient.

    Returns:
        Path to the generated output file.
    """
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")

    work_dir = video_path.parent

    # Stage 1: Extract landmarks
    print("Stage 1: Extracting landmarks...")
    npz_path = do_extract(video_path, work_dir / "temp_extract.npz")

    # Stage 2: Smooth
    print("Stage 2: Smoothing landmarks...")
    smooth_path = do_smooth(npz_path, work_dir / "temp_smooth.npz", min_cutoff, beta)

    # Stage 3: Solve rotations
    print("Stage 3: Solving rotations...")
    rot_path = do_solve(smooth_path, work_dir / "temp_rotations.npz")

    # Stage 4: Retarget to Mixamo
    print("Stage 4: Retargeting to Mixamo skeleton...")
    data = dict(np.load(rot_path, allow_pickle=True))
    internal_rotations = {}
    for key in data:
        if key.startswith("rot_"):
            bone_name = key[4:]
            internal_rotations[bone_name] = data[key]

    result = retarget(internal_rotations, num_frames=int(data["num_frames"]))
    mixamo_rotations = result["rotations"]

    # Stage 5: Export
    print(f"Stage 5: Exporting as {format.upper()}...")
    write_gltf(
        mixamo_rotations,
        fps=float(data["fps"]),
        output_path=output_path,
        glb=(format == "glb"),
        bone_names=result["bone_names"],
        hierarchy=result["hierarchy"],
    )

    # Cleanup temp files
    for p in [npz_path, smooth_path, rot_path]:
        p.unlink(missing_ok=True)

    print(f"Done! Output: {output_path}")
    return output_path
