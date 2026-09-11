"""CLI interface — thin wrapper around mimic.pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="mimic", help="Convert MP4 video to 3D glTF/GLB animation.")


@app.command()
def convert(
    video: Path = typer.Argument(..., help="Path to input MP4 video file."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file path."),
    fps: int | None = typer.Option(None, "--fps", help="Target frame rate."),
    format: str = typer.Option("glb", "-f", "--format", help="Output format: glb or gltf."),
) -> None:
    """Convert a video of a person moving into a 3D glTF/GLB animation."""
    from mimic.pipeline import run

    result = run(video_path=video, output_path=output, fps=fps, format=format)
    typer.echo(f"Output written to: {result}")


@app.command()
def info() -> None:
    """Show project information."""
    import mimic

    typer.echo(f"Mimic v{mimic.__version__}")


if __name__ == "__main__":
    app()
