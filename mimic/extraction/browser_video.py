"""Publish browser-playable H.264 overlays without leaving encoding scratch files."""
import subprocess
from pathlib import Path

from mimic.artifacts import atomic_output
from mimic.errors import MimicError


def encode_browser_video(path):
    """Replace a temporary OpenCV MP4 only after a successful H.264 encode."""
    import imageio_ffmpeg

    path = Path(path)
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        with atomic_output(path) as encoded:
            subprocess.run(
                [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                 '-i', str(path), '-map', '0:v:0', '-an', '-c:v', 'libx264',
                 '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                 '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart',
                 str(encoded)],
                check=True, capture_output=True, timeout=300,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise MimicError('video_encode',
                         'Could not encode a browser-compatible overlay. '
                         'Check the FFmpeg installation and available disk space.') from exc
