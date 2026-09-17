from concurrent.futures import Future
import json

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from interfaces.api.main import create_app
from interfaces.api.jobs import convert_job, read_record, write_record


class Queue:
    def __init__(self):
        self.calls = []

    def submit(self, *args):
        future = Future()
        self.calls.append((args, future))
        return future

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def api(tmp_path):
    queue = Queue()
    with TestClient(create_app(tmp_path, capacity=1, max_upload_bytes=8,
                              executor_factory=lambda: queue)) as client:
        yield client, queue, tmp_path


def upload(client, body=b'video', **params):
    return client.post('/api/jobs', content=body, headers={'Content-Type': 'video/mp4'},
                       params=params)


def test_limits_and_validation_release_capacity(api):
    client, queue, root = api
    assert upload(client, b'123456789').status_code == 413
    assert upload(client, b'').status_code == 400
    assert upload(client, format='fbx').status_code == 422
    assert client.post('/api/jobs', content=b'video').status_code == 415
    assert list(root.iterdir()) == []
    assert upload(client).status_code == 202
    assert upload(client).status_code == 429
    assert client.get('/api/health').status_code == 200


def test_status_download_and_delete(api):
    client, queue, root = api
    job = upload(client, format='bvh', overlay=True).json()['id']
    assert client.get(f'/api/jobs/{job}').json()['status'] == 'queued'
    assert client.delete(f'/api/jobs/{job}').status_code == 409
    assert client.get(f'/api/jobs/{job}/files/animation').status_code == 409
    folder = root / job
    (folder / 'animation.bvh').write_bytes(b'animation')
    (folder / 'animation_overlay.mp4').write_bytes(b'overlay')
    write_record(folder, **dict(read_record(folder), status='complete'))
    queue.calls[0][1].set_result(None)
    assert client.get(f'/api/jobs/{job}/files/animation').content == b'animation'
    assert client.get(f'/api/jobs/{job}/files/overlay').content == b'overlay'
    assert client.get(f'/api/jobs/{job}/files/input.mp4').status_code == 422
    assert client.delete(f'/api/jobs/{job}').status_code == 204
    assert client.get(f'/api/jobs/{job}').status_code == 404
    assert not folder.exists()


def test_worker_crash_releases_slot(api):
    client, queue, root = api
    job = upload(client).json()['id']
    queue.calls[0][1].set_exception(RuntimeError('crashed'))
    assert client.get(f'/api/jobs/{job}').json()['code'] == 'worker_failed'
    assert not (root / job / 'input.mp4').exists()
    assert upload(client).status_code == 202


def test_restart_marks_abandoned_jobs_failed(tmp_path):
    folder = tmp_path / 'old'
    folder.mkdir()
    write_record(folder, status='running')
    (folder / 'input.mp4').write_bytes(b'video')
    with TestClient(create_app(tmp_path, executor_factory=Queue)):
        assert read_record(folder)['code'] == 'server_restarted'
        assert not (folder / 'input.mp4').exists()


def test_conversion_failure_removes_upload(tmp_path, monkeypatch):
    from mimic import pipeline
    from mimic.errors import MimicError
    write_record(tmp_path, status='queued')
    (tmp_path / 'input.mp4').write_bytes(b'video')

    def fail(*args, **kwargs):
        raise MimicError('no_human_pose', 'No person detected')

    monkeypatch.setattr(pipeline, 'run', fail)
    convert_job(str(tmp_path), {'format': 'bvh', 'overlay': False})
    assert read_record(tmp_path)['code'] == 'no_human_pose'
    assert not (tmp_path / 'input.mp4').exists()


def test_stage_updates_are_published(tmp_path):
    from mimic.artifacts import run_report
    path = tmp_path / 'status.json'
    with run_report(path) as report:
        report['stage'] = 'smoothing'
        assert json.loads(path.read_text())['stage'] == 'smoothing'


def test_chunked_upload_limit_cleans_partial_file(api):
    client, queue, root = api
    response = client.post('/api/jobs', content=iter([b'12345', b'67890']),
                           headers={'Content-Type': 'video/mp4'})
    assert response.status_code == 413
    assert list(root.iterdir()) == []
    assert upload(client).status_code == 202


def test_real_process_reports_invalid_video(tmp_path):
    import time
    with TestClient(create_app(tmp_path)) as client:
        response = upload(client, b'not a video', format='bvh')
        assert response.status_code == 202
        job = response.json()['id']
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = client.get(f'/api/jobs/{job}').json()
            if result['status'] == 'failed':
                break
            time.sleep(0.1)
        assert result['status'] == 'failed'
        assert result['code'] != 'worker_failed', result
    assert not (tmp_path / job / 'input.mp4').exists()
