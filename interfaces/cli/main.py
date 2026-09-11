"""CLI interface — thin wrapper around mimic.pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="mimic", help="Convert MP4 video to 3D glTF/GLB animation.")


@app.command()
def extract(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output .npz file path."),
    visualize: bool = typer.Option(
        False, "-v", "--visualize", help="Save 3D scatter plots of landmarks."
    ),
) -> None:
    """Phase 1: Extract pose landmarks from a video to .npz."""
    from mimic.extraction.phase1 import extract as do_extract

    result = do_extract(video_path=video, output_path=output)

    if visualize:
        import numpy as np

        from mimic.extraction.visualize import plot_landmarks_all_frames

        data = dict(np.load(result, allow_pickle=True))
        vis_dir = result.parent / "visualize"
        saved = plot_landmarks_all_frames(
            data["world_landmarks"],
            output_dir=vis_dir,
            visibility=data["visibility"],
        )
        typer.echo(f"Saved {len(saved)} visualizations to: {vis_dir}")


@app.command()
def smooth(
    npz: Path = typer.Argument(..., help="Path to extracted landmarks .npz file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output .npz file path."),
    min_cutoff: float = typer.Option(
        1.0, "--min-cutoff", help="One-Euro min cutoff (higher=more smooth)."
    ),
    beta: float = typer.Option(0.007, "--beta", help="One-Euro speed coefficient."),
) -> None:
    """Phase 2: Smooth landmark trajectories to remove jitter."""
    from mimic.processing.phase2 import smooth as do_smooth

    result = do_smooth(npz_path=npz, output_path=output, min_cutoff=min_cutoff, beta=beta)
    typer.echo(f"Smoothed landmarks saved to: {result}")


@app.command()
def convert(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path."),
    fps: Optional[int] = typer.Option(None, "--fps", help="Target frame rate."),
    format: str = typer.Option("glb", "-f", "--format", help="Output format: glb or gltf."),
) -> None:
    """Convert a video of a person moving into a 3D glTF/GLB animation."""
    from mimic.pipeline import run

    result = run(video_path=video, output_path=output, fps=fps, format=format)
    typer.echo(f"Output written to: {result}")


@app.command()
def visualize(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    npz: Optional[Path] = typer.Option(None, "-n", "--npz", help="Path to landmarks .npz file."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output video path."),
    threshold: float = typer.Option(
        0.3, "-t", "--threshold", help="Minimum visibility to draw a landmark."
    ),
) -> None:
    """Generate a video with the skeleton drawn over the original footage."""
    from mimic.extraction.overlay import generate_overlay_video

    if npz is None:
        npz = video.with_suffix(".npz")
    result = generate_overlay_video(video, npz, output, threshold=threshold)
    typer.echo(f"Overlay video saved to: {result}")


@app.command()
def merge(
    model: Path = typer.Argument(..., help="Path to Mixamo rigged .glb model (T-pose)."),
    animation: Path = typer.Argument(..., help="Path to skeleton-only animation .glb."),
    output: Path = typer.Option(..., "-o", "--output", help="Output animated .glb path."),
) -> None:
    """Merge a skeleton animation onto a full rigged Mixamo model."""
    from mimic.export.merge_animation import merge_animation

    result = merge_animation(model, animation, output)
    typer.echo(f"Animated model saved to: {result}")


@app.command()
def info() -> None:
    """Show project information."""
    import mimic

    typer.echo(f"Mimic v{mimic.__version__}")


if __name__ == "__main__":
    app()
