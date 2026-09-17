"""Validated file access and atomic publication of generated artifacts."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile
import numpy as np
from mimic.errors import MimicError


def output_path(path, suffix, inputs=()):
    path = Path(path)
    if path.suffix.lower() != suffix:
        raise MimicError('output_format', f'Output must end in {suffix}: {path}')
    if path.is_dir() or any(path.resolve() == Path(p).resolve() for p in inputs):
        raise MimicError('output_conflict', f'Output must not replace an input file or directory: {path}')
    return path


@contextmanager
def atomic_output(path):
    """Keep any prior output intact until the new file has been written fully."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix='.' + path.stem + '-', suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
    try:
        yield temporary
        if not temporary.stat().st_size:
            raise MimicError('empty_output', f'Writer produced an empty file: {path}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz(path, **data):
    with atomic_output(path) as temp:
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **data)


def load_npz(path):
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 512 * 1024**2:
                raise MimicError('archive_too_large', 'Decoded NPZ data exceeds 512 MiB.')
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (zipfile.BadZipFile, EOFError, ValueError) as exc:
        if isinstance(exc, MimicError):
            raise
        raise MimicError('invalid_archive', f'Cannot read numeric NPZ data from {path}: {exc}') from exc

@contextmanager
def run_report(path):
    """Record current-run status so retained checkpoints cannot imply success."""
    import json
    from datetime import datetime, timezone
    class Report(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == 'stage':
                publish()
    report = Report(status='running', stage='extraction',
                    started_at=datetime.now(timezone.utc).isoformat())
    def publish():
        with atomic_output(path) as temporary:
            temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    publish()
    try:
        yield report
    except BaseException as exc:
        report.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=str(exc), code=getattr(exc, 'code', type(exc).__name__))
        publish()
        raise
    else:
        report['status'] = 'complete'
        publish()
