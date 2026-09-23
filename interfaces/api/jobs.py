"""Disk-backed job records and an isolated conversion worker."""
from __future__ import annotations

import json
import time
from pathlib import Path

from mimic.artifacts import atomic_output


def write_record(folder, **record):
    with atomic_output(folder / 'job.json') as path:
        path.write_text(json.dumps(record), encoding='utf-8')


def read_record(folder):
    return json.loads((folder / 'job.json').read_text(encoding='utf-8'))


def convert_job(directory, options):
    """Executed in a process, never in the HTTP event loop."""
    from mimic.pipeline import run
    from mimic.config import MODELS_DIR

    folder = Path(directory)
    record = read_record(folder)
    write_record(folder, **dict(record, status='running'))
    started = time.perf_counter()
    try:
        run(folder / 'input.mp4', folder / ('animation.' + options['format']),
            model_path=MODELS_DIR / 'model.glb' if options['format'] == 'glb' else None,
            **options)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        write_record(folder, **dict(record, status='failed',
                     code=getattr(exc, 'code', 'conversion_failed'), error=str(exc),
                     duration_ms=duration_ms))
    else:
        duration_ms = int((time.perf_counter() - started) * 1000)
        print(f'Conversion job {folder.name} completed in {duration_ms} ms', flush=True)
        write_record(folder, **dict(record, status='complete', duration_ms=duration_ms))
    finally:
        (folder / 'input.mp4').unlink(missing_ok=True)
