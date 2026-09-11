"""Pipeline orchestrator — the single entrypoint for CLI and future API."""

from __future__ import annotations

from pathlib import Path

from mimic.export.gltf_writer import write_gltf
from mimic.extraction.pose_extractor import extract_landmarks
from mimic.extraction.video_reader import read_frames
from mimic.processing.rotation_solver import solve_rotations
from mimic.processing.smoothing import smooth_landmarks
from mimic.retargeting.retarget import retarget


def run(
    video_path: Path,
    output_path: Path | None = None,
    fps: int | None = None,
    format: str = "glb",
) -> Path:
    """Run the full video-to-3D-animation pipeline.

    Args:
        video_path: Path to the input MP4 video.
        output_path: Path for the output file. Defaults to video name with .glb extension.
        fps: Target frame rate. None keeps source fps.
        format: Output format — "glb" (default) or "gltf".

    Returns:
        Path to the generated output file.
    """
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")

    # Stage 1: Read video frames
    frames, source_fps = read_frames(video_path)

    # Stage 2: Extract pose landmarks
    landmarks = extract_landmarks(frames)

    # Stage 3: Smooth landmark trajectories
    smoothed = smooth_landmarks(landmarks, fps=source_fps)

    # Stage 4: Solve bone rotations from landmarks
    rotations = solve_rotations(smoothed)

    # Stage 5: Retarget onto Mixamo skeleton
    mixamo_rotations = retarget(rotations)

    # Stage 6: Export
    write_gltf(mixamo_rotations, fps=source_fps, output_path=output_path, glb=(format == "glb"))

    return output_path
