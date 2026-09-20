import subprocess

import cv2
import numpy as np
import pytest

from mimic.errors import MimicError
from mimic.extraction.browser_video import encode_browser_video


def test_h264_overlay_preserves_frames_and_cleans_scratch(tmp_path):
    path = tmp_path / 'overlay.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (64, 64))
    assert writer.isOpened()
    for i in range(24):
        writer.write(np.full((64, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    encode_browser_video(path)
    video = cv2.VideoCapture(str(path))
    try:
        assert video.isOpened()
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        assert ''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4)) in {'avc1', 'h264'}
        assert video.read()[0]
    finally:
        video.release()
    data = path.read_bytes()
    assert data.index(b'moov') < data.index(b'mdat')
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('error', [subprocess.CalledProcessError(1, 'ffmpeg'), subprocess.TimeoutExpired('ffmpeg', 300)])
def test_encode_failure_preserves_original_and_cleans_scratch(tmp_path, monkeypatch, error):
    path = tmp_path / 'overlay.mp4'
    path.write_bytes(b'original')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, 'run', fail)
    monkeypatch.setattr('imageio_ffmpeg.get_ffmpeg_exe', lambda: 'ffmpeg')
    with pytest.raises(MimicError, match='video_encode'):
        encode_browser_video(path)
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]
