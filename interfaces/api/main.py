"""Local single-server API: streamed uploads and a bounded process queue."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore
from typing import Literal, Optional
from uuid import UUID, uuid4

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from mimic.config import OUTPUT_DIR, VIDEO_LIMITS
from interfaces.api.jobs import convert_job, read_record, write_record


def create_app(root: Optional[Path] = None, capacity: int = 4,
               max_upload_bytes: int = VIDEO_LIMITS.max_file_bytes,
               executor_factory=None):
    root = (root or OUTPUT_DIR / 'jobs').resolve()
    slots = BoundedSemaphore(capacity)

    def remove(folder):
        # Only generated job directories immediately inside our configured root.
        if folder.resolve().parent != root or folder.is_symlink():
            raise ValueError('Invalid job directory')
        shutil.rmtree(folder)

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        for path in root.glob('*/job.json'):
            record = read_record(path.parent)
            if record['status'] in ('queued', 'running', 'uploading'):
                write_record(path.parent, **dict(record, status='failed',
                             code='server_restarted', error='Server stopped before completion.'))
                (path.parent / 'input.mp4').unlink(missing_ok=True)
        app.state.executor = (executor_factory() if executor_factory else
                              ProcessPoolExecutor(max_workers=1,
                                  mp_context=multiprocessing.get_context('spawn')))
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.executor.shutdown, wait=True)

    app = FastAPI(title='Mimic API', version='0.1.0', lifespan=lifespan)

    def folder_for(job_id: UUID):
        folder = root / str(job_id)
        if not (folder / 'job.json').is_file():
            raise HTTPException(404, 'Job not found')
        return folder

    @app.get('/api/health')
    def health():
        return {'status': 'ok'}

    @app.post('/api/jobs', status_code=202, openapi_extra={
        'requestBody': {'required': True, 'content': {
            'video/mp4': {'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def submit(request: Request, format: Literal['glb', 'bvh'] = 'glb',
                     overlay: bool = False,
                     max_missing_frames: int = Query(20, ge=0, le=14400),
                     fps: Optional[int] = Query(None, ge=1, le=240)):
        if request.headers.get('content-type', '').split(';')[0] not in (
                'video/mp4', 'application/octet-stream'):
            raise HTTPException(415, 'Send raw MP4 bytes as video/mp4; not multipart form data')
        length = request.headers.get('content-length')
        if length:
            try:
                size = int(length)
            except ValueError:
                raise HTTPException(400, 'Invalid Content-Length')
            if size < 0:
                raise HTTPException(400, 'Invalid Content-Length')
            if size > max_upload_bytes:
                raise HTTPException(413, 'Video exceeds upload size limit')
        if not slots.acquire(blocking=False):
            raise HTTPException(429, 'Conversion queue is full', headers={'Retry-After': '5'})
        folder = root / str(uuid4())
        submitted = False
        try:
            folder.mkdir()
            record = {'id': folder.name, 'status': 'uploading', 'format': format,
                      'overlay': overlay}
            write_record(folder, **record)
            size = 0
            async with await anyio.open_file(folder / 'input.mp4', 'wb') as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, 'Video exceeds upload size limit')
                    await stream.write(chunk)
            if not size:
                raise HTTPException(400, 'Video is empty')
            record['status'] = 'queued'
            write_record(folder, **record)
            options = dict(format=format, overlay=overlay,
                           max_missing_frames=max_missing_frames, fps=fps)
            try:
                future = app.state.executor.submit(convert_job, str(folder), options)
            except RuntimeError as exc:
                raise HTTPException(503, 'Conversion worker unavailable; restart the server') from exc

            def finished(result):
                try:
                    result.result()
                except BaseException:
                    write_record(folder, **dict(record, status='failed',
                                 code='worker_failed', error='Conversion worker stopped unexpectedly.'))
                    (folder / 'input.mp4').unlink(missing_ok=True)
                finally:
                    slots.release()

            future.add_done_callback(finished)
            submitted = True
            return {'id': folder.name, 'status': 'queued',
                    'status_url': f'/api/jobs/{folder.name}'}
        finally:
            if not submitted:
                if folder.exists():
                    await asyncio.to_thread(remove, folder)
                slots.release()

    @app.get('/api/jobs/{job_id}')
    def status(job_id: UUID):
        folder = folder_for(job_id)
        record = read_record(folder)
        report = folder / 'animation_status.json'
        if report.exists():
            details = json.loads(report.read_text(encoding='utf-8'))
            record['progress'] = {k: v for k, v in details.items()
                                  if k not in ('output', 'overlay', 'status')}
        if record['status'] == 'complete':
            record['files'] = {'animation': f'/api/jobs/{job_id}/files/animation'}
            if record['overlay']:
                record['files']['overlay'] = f'/api/jobs/{job_id}/files/overlay'
        return record

    @app.get('/api/jobs/{job_id}/files/{kind}')
    def download(job_id: UUID, kind: Literal['animation', 'overlay']):
        folder = folder_for(job_id)
        record = read_record(folder)
        if record['status'] != 'complete':
            raise HTTPException(409, 'Job has not completed successfully')
        name = 'animation.' + record['format'] if kind == 'animation' else 'animation_overlay.mp4'
        path = folder / name
        if not path.is_file():
            raise HTTPException(404, 'Output not found')
        media = ('video/mp4' if kind == 'overlay' else
                 'model/gltf-binary' if record['format'] == 'glb' else 'application/octet-stream')
        return FileResponse(path, media_type=media, filename=name)

    @app.delete('/api/jobs/{job_id}', status_code=204)
    def delete(job_id: UUID):
        folder = folder_for(job_id)
        if read_record(folder)['status'] not in ('complete', 'failed'):
            raise HTTPException(409, 'Cannot delete an active job')
        remove(folder)

    return app


app = create_app()
