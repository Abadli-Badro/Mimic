"""CLI interface — thin wrapper around mimic.pipeline."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file, MAX_MISSING_BONE_FRAMES
from typing import Optional

import typer
import sys
import traceback
from functools import wraps
from mimic.errors import MimicError


def cli_errors(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (MimicError, OSError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from None
        except Exception as exc:
            # Keep unexpected failures visible even when Typer/Rich's hook fails.
            typer.echo("Unexpected error (plain traceback follows):", err=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            raise typer.Exit(code=1) from None
    return wrapped


app = typer.Typer(
    name="mimic",
    help=("Convert a video into an in-place BVH or GLB/glTF skeleton animation. "
          "Use merge to apply a GLB animation to a rigged character. "
          "Generated files default to the project-root output/ directory."),
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=("Quick start: mimic convert video.mp4 | "
            "Character: mimic merge data/models/model.glb output/video.glb | "
            "Command help: mimic COMMAND --help"),
)


@app.command("run")
@cli_errors
def run_pipeline(
    video: Path = typer.Argument(..., help="Input video path."),
    format: str = typer.Option("glb", "-f", "--format", help="Final format: glb or bvh."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Final animation path; defaults under output/."),
    overlay: bool = typer.Option(False, "--overlay", help="Also save a landmark overlay of the original video beside the animation."),
    max_missing_frames: int = typer.Option(MAX_MISSING_BONE_FRAMES, "--max-missing-frames", min=0,
        help="Maximum unseen source frames per bone before ending the animation."),
    fps: Optional[int] = typer.Option(None, "--fps", min=1, max=240, help="Resample animation; overlay keeps source FPS."),
    model: Optional[Path] = typer.Option(None, "--model", help="GLB character model; default: data/models/model.glb. Not used for BVH."),
) -> None:
    """Run the entire pipeline and export an animated character GLB or skeleton BVH.

    Example: mimic run "data/input/Dance Reference.mp4" --format glb --overlay --max-missing-frames 20

    Extracts once, smooths, solves rotations, exports, and merges the model for GLB.
    Temporary intermediate files are deleted on exit. A status JSON remains beside the export.
    """
    from mimic.config import MODELS_DIR
    from mimic.pipeline import run

    if format not in {'glb', 'bvh'}:
        raise typer.BadParameter('Choose glb or bvh.', param_hint='--format')
    if format == 'bvh' and model is not None:
        raise typer.BadParameter('--model applies only to GLB output.', param_hint='--model')
    target_model = (model or MODELS_DIR / 'model.glb') if format == 'glb' else None
    result = run(video_path=video, output_path=output, fps=fps, format=format,
                 max_missing_frames=max_missing_frames, model_path=target_model, overlay=overlay)
    typer.echo(f"Animation saved to: {result}")


@app.command()
@cli_errors
def extract(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output NPZ path (default: output/<video>.npz)."),
    visualize: bool = typer.Option(
        False, "-v", "--visualize", help="Save 3D landmark plots to output/previews/<video>/."
    ),
) -> None:
    """Extract pose landmarks from a video to NPZ.

    Example: mimic extract video.mp4 --visualize
    """
    from mimic.extraction.video_to_landmarks import extract as do_extract

    result = do_extract(video_path=video, output_path=output)

    if visualize:
        import numpy as np

        from mimic.extraction.landmark_plots import plot_landmarks_all_frames

        data = dict(np.load(result, allow_pickle=False))
        vis_dir = output_file("previews") / video.stem
        saved = plot_landmarks_all_frames(
            data["world_landmarks"],
            output_dir=vis_dir,
            visibility=data["visibility"],
        )
        typer.echo(f"Saved {len(saved)} visualizations to: {vis_dir}")


@app.command()
@cli_errors
def smooth(
    npz: Path = typer.Argument(..., help="Path to extracted landmarks .npz file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output NPZ path (default: output/<name>_smooth.npz)."),
    min_cutoff: float = typer.Option(
        1.0, "--min-cutoff", help="One-Euro min cutoff (lower=more smooth)."
    ),
    beta: float = typer.Option(0.007, "--beta", help="Motion response: higher values reduce lag during fast movement."),
) -> None:
    """Smooth extracted landmark trajectories to reduce jitter.

    Example: mimic smooth output/video.npz --min-cutoff 0.5
    """
    from mimic.processing.smooth_landmarks import smooth as do_smooth

    result = do_smooth(npz_path=npz, output_path=output, min_cutoff=min_cutoff, beta=beta)
    typer.echo(f"Smoothed landmarks saved to: {result}")


@app.command()
@cli_errors
def convert(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path (default: output/<video>.<format>). Use --format to select the format."),
    fps: Optional[int] = typer.Option(None, "--fps", help="Resample to this frame rate; omitted keeps the source rate."),
    max_missing_frames: int = typer.Option(
        MAX_MISSING_BONE_FRAMES, "--max-missing-frames", min=0,
        help="Maximum consecutive unseen source frames per bone; first timeout ends the clip.",
    ),
    format: str = typer.Option("glb", "-f", "--format", help="Output format: glb, gltf, or bvh."),
) -> None:
    """Convert a video into a skeleton animation (GLB, glTF, or BVH).

    Examples: mimic convert video.mp4 --fps 30

    mimic convert video.mp4 --format bvh

    Intermediate files are isolated per run and deleted on exit.
    GLB output contains a skeleton; use merge to add a rigged character.
    """
    from mimic.pipeline import run

    result = run(video_path=video, output_path=output, fps=fps, format=format, max_missing_frames=max_missing_frames)
    typer.echo(f"Output written to: {result}")


@app.command()
@cli_errors
def visualize(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    npz: Optional[Path] = typer.Option(None, "-n", "--npz", help="Landmark NPZ path (default: output/<video>.npz, created by extract)."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Overlay path (default: output/<video>_overlay.mp4)."),
    threshold: float = typer.Option(
        0.3, "-t", "--threshold", help="Minimum landmark visibility from 0 to 1; higher hides uncertain points."
    ),
) -> None:
    """Draw detected landmarks over the original video.

    Example: mimic visualize video.mp4

    Run extract first, or pass --npz with a saved landmark file.
    For an overlay during conversion, use mimic run --overlay.
    """
    from mimic.extraction.overlay import generate_overlay_video

    if npz is None:
        npz = output_file(video.stem + ".npz")
    result = generate_overlay_video(video, npz, output, threshold=threshold)
    typer.echo(f"Overlay video saved to: {result}")


@app.command()
@cli_errors
def merge(
    model: Path = typer.Argument(..., help="Path to Mixamo rigged .glb model (T-pose)."),
    animation: Path = typer.Argument(..., help="Path to skeleton-only animation .glb."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output GLB path (default: output/<animation>_animated.glb)."),
) -> None:
    """Apply a skeleton GLB animation to a rigged Mixamo GLB model.

    Example: mimic merge data/models/model.glb output/video.glb
    """
    from mimic.export.merge_animation import merge_animation

    result = merge_animation(model, animation, output)
    typer.echo(f"Animated model saved to: {result}")


@app.command()
@cli_errors
def info() -> None:
    """Show project information."""
    import mimic

    typer.echo(f"Mimic v{mimic.__version__}")


if __name__ == "__main__":
    app()
