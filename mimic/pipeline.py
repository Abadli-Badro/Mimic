"""Pipeline orchestrator — the single entrypoint for CLI and future API."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from mimic.config import output_file, MAX_MISSING_BONE_FRAMES
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
    max_missing_frames: int = MAX_MISSING_BONE_FRAMES,
    bone_limits: dict[str, int] | None = None,
    model_path: Path | None = None,
    overlay: bool = False,
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
    if model_path is not None:
        if format != 'glb':
            raise ValueError('Character model merging requires GLB output.')
        if not model_path.is_file():
            raise FileNotFoundError(f'Character model not found: {model_path}')
    if output_path is None:
        suffix = '_animated' if model_path is not None else ''
        output_path = output_file(f"{video_path.stem}{suffix}.{format}")

    output_path = check_output(output_path, '.' + format, [video_path] + ([model_path] if model_path else []))
    number(min_cutoff, 'min_cutoff')
    number(beta, 'beta', inclusive=True)
    if fps is not None:
        number(fps, 'fps', maximum=240)
    work_root = output_file("intermediate")
    work_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="run-", dir=work_root) as temporary, run_report(
        output_path.with_name(output_path.stem + '_status.json')
    ) as report:
        work_dir = Path(temporary)
        # Extract landmarks
        print("Extracting landmarks...")
        npz_path = do_extract(video_path, work_dir / "landmarks.npz")

        # Smooth
        report["stage"] = "smoothing"
        print("Smoothing landmarks...")
        smooth_path = do_smooth(npz_path, work_dir / "landmarks_smooth.npz", min_cutoff, beta, max_missing_frames, bone_limits)

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
        report['exported_frames'] = int(data['num_frames'])
        report['source_frames'] = int(data['num_frames'])
        report['first_frame'] = int(data.get('first_frame', 0))
        report['source_fps'] = float(data['fps'])
        report['stopped_bones'] = list(data.get('stopped_bones', []))
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
        report['fps'] = output_fps
        report['exported_frames'] = len(next(iter(internal_rotations.values())))

        report["stage"] = "export"
        print(f"Exporting as {format.upper()}...")
        if format == "bvh":
            write_bvh(internal_rotations, output_fps, output_path)
        else:
            result = retarget(internal_rotations)
            skeleton_path = work_dir / 'skeleton.glb' if model_path is not None else output_path
            write_gltf(result["rotations"], fps=output_fps, output_path=skeleton_path,
                       glb=(format == "glb"), bone_names=result["bone_names"],
                       hierarchy=result["hierarchy"])

        if model_path is not None:
            from mimic.export.merge_animation import merge_animation
            report['stage'] = 'model merge'
            merge_animation(model_path, skeleton_path, output_path)
        if overlay:
            from mimic.extraction.overlay import generate_overlay_video
            report['stage'] = 'overlay export'
            overlay_path = output_path.with_name(output_path.stem + '_overlay.mp4')
            generate_overlay_video(video_path, npz_path, overlay_path)
            report['overlay'] = str(overlay_path)
            print(f"Overlay: {overlay_path}")
        report['output'] = str(output_path)
        print(f"Done! Output: {output_path}")
        return output_path
