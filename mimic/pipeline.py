"""Pipeline orchestrator — the single entrypoint for CLI and future API."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file
from mimic.errors import stage
from mimic.artifacts import run_report, load_npz, output_path as check_output
from mimic.validation import rotations as validate_rotations, number

import numpy as np

from mimic.export.gltf_writer import write_gltf
from mimic.export.bvh_writer import write_bvh
from scipy.spatial.transform import Rotation, Slerp
from mimic.extraction.video_to_landmarks import extract as do_extract
from mimic.processing.smooth_landmarks import smooth as do_smooth
from mimic.processing.landmarks_to_rotations import solve as do_solve
from mimic.retargeting.retarget import retarget


@stage("conversion")
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
        output_path: Path for the output file. Defaults to output/<video name>.<format>.
        fps: Target frame rate. None keeps source fps.
        format: Output format — "glb" (default) or "gltf".
        min_cutoff: One-Euro filter min cutoff.
        beta: One-Euro filter speed coefficient.

    Returns:
        Path to the generated output file.
    """
    if format not in {"glb", "gltf", "bvh"}:
        raise ValueError("Format must be glb, gltf, or bvh")
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("Target fps must be positive")
    if output_path is None:
        output_path = output_file(f"{video_path.stem}.{format}")

    output_path = check_output(output_path, '.' + format, [video_path])
    number(min_cutoff, 'min_cutoff')
    number(beta, 'beta', inclusive=True)
    if fps is not None:
        number(fps, 'fps', maximum=240)
    work_dir = output_file("intermediate") / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    with run_report(work_dir / "status.json") as report:
        # Extract landmarks
        print("Extracting landmarks...")
        npz_path = do_extract(video_path, work_dir / "landmarks.npz")

        # Smooth
        report["stage"] = "smoothing"
        print("Smoothing landmarks...")
        smooth_path = do_smooth(npz_path, work_dir / "landmarks_smooth.npz", min_cutoff, beta)

        # Solve rotations
        report["stage"] = "rotation solving"
        print("Solving rotations...")
        rot_path = do_solve(smooth_path, work_dir / "rotations.npz")

        # Retarget to Mixamo
        report["stage"] = "retargeting"
        print("Retargeting to Mixamo skeleton...")
        data = load_npz(rot_path)
        internal_rotations = {}
        for key in data:
            if key.startswith("rot_"):
                bone_name = key[4:]
                internal_rotations[bone_name] = data[key]

        from mimic.processing.rotation_solver import BONE_HIERARCHY
        validate_rotations(internal_rotations, BONE_HIERARCHY, data.get('num_frames'))
        if 'fps' not in data or 'num_frames' not in data:
            raise ValueError('Rotation file must contain fps and num_frames.')
        output_fps = number(data['fps'], 'fps', maximum=240)
        num_frames = int(data["num_frames"])
        if fps is not None and fps != output_fps and num_frames > 1:
            source_times = np.arange(num_frames) / output_fps
            target_times = np.arange(int(np.floor(source_times[-1] * fps)) + 1) / fps
            internal_rotations = {
                name: Slerp(source_times, Rotation.from_quat(quats))(target_times).as_quat()
                for name, quats in internal_rotations.items()
            }
        if fps is not None:
            output_fps = float(fps)

        report["stage"] = "export"
        print(f"Exporting as {format.upper()}...")
        if format == "bvh":
            write_bvh(internal_rotations, output_fps, output_path)
        else:
            result = retarget(internal_rotations)
            write_gltf(result["rotations"], fps=output_fps, output_path=output_path,
                       glb=(format == "glb"), bone_names=result["bone_names"],
                       hierarchy=result["hierarchy"])

        print(f"Done! Output: {output_path}")
        return output_path
