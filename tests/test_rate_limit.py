import pytest

pytest.importorskip('fastapi')
from fastapi import FastAPI
from fastapi.testclient import TestClient
from interfaces.api.rate_limit import RateLimitMiddleware


def test_window_reset_and_bounded_storage():
    time = [0.0]
    limiter = RateLimitMiddleware(None, max_clients=2, clock=lambda: time[0])
    assert limiter.check('a', 1) == 0
    assert limiter.check('a', 1) == 60
    assert limiter.check('b', 1) == 0
    assert limiter.check('c', 1) == 60
    assert len(limiter.buckets) == 2
    time[0] = 60
    assert limiter.check('c', 1) == 0
    assert len(limiter.buckets) == 1


def test_upload_polling_and_peer_isolation():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, uploads=1, requests=2)

    @app.api_route('/api/jobs', methods=['GET', 'POST'])
    def jobs():
        return {'ok': True}

    with TestClient(app, client=('192.0.2.1', 123)) as client:
        assert client.post('/api/jobs').status_code == 200
        response = client.post('/api/jobs', headers={'X-Forwarded-For': '192.0.2.9'})
        assert response.status_code == 429
        assert 1 <= int(response.headers['Retry-After']) <= 60
        assert response.json()['code'] == 'rate_limited'
        assert client.get('/api/jobs').status_code == 200
        assert client.get('/api/jobs').status_code == 200
        assert client.get('/api/jobs').status_code == 429
        assert client.get('/api/health').status_code == 404  # Exempt, reaches router.
        assert client.options('/api/jobs').status_code != 429
    with TestClient(app, client=('192.0.2.2', 123)) as client:
        assert client.post('/api/jobs').status_code == 200
