"""Scratch directories must be isolated and removed even on interrupted runs."""
from pathlib import Path

import pytest

from mimic import pipeline


@pytest.mark.parametrize('error', [ValueError('bad video'), KeyboardInterrupt()])
def test_cleanup_preserves_unrelated_files(tmp_path, monkeypatch, error):
    monkeypatch.setattr(pipeline, 'output_file', lambda name: tmp_path / name)
    root = tmp_path / 'intermediate'
    root.mkdir()
    unrelated = root / 'existing.npz'
    unrelated.write_bytes(b'keep')
    seen = []

    def extract(video, path):
        seen.append(path.parent)
        path.write_bytes(b'partial extraction')
        raise error

    monkeypatch.setattr(pipeline, 'do_extract', extract)
    for name in ('one', 'two'):
        with pytest.raises((ValueError, KeyboardInterrupt)):
            pipeline.run(Path('same.mp4'), tmp_path / f'{name}.glb')
    assert seen[0] != seen[1]
    assert all(not path.exists() for path in seen)
    assert unrelated.read_bytes() == b'keep'


def test_overlapping_runs_have_separate_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'output_file', lambda name: tmp_path / name)
    paths = []

    def extract(video, path):
        paths.append(path)
        path.write_bytes(b'raw')
        if len(paths) == 1:
            with pytest.raises(ValueError):
                pipeline.run(video, tmp_path / 'inner.glb')
            assert path.exists()
            assert not paths[1].exists()
        raise ValueError('stop after extraction')

    monkeypatch.setattr(pipeline, 'do_extract', extract)
    with pytest.raises(ValueError):
        pipeline.run(Path('same.mp4'), tmp_path / 'outer.glb')
    assert paths[0].parent != paths[1].parent
    assert list((tmp_path / 'intermediate').iterdir()) == []
