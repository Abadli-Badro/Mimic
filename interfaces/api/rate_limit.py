"""Bounded, per-client fixed-window limits for the single-process API."""
from collections import OrderedDict
from math import ceil
from threading import Lock
from time import monotonic

from starlette.responses import JSONResponse


class RateLimitMiddleware:
    def __init__(self, app, uploads=10, requests=120, window=60,
                 max_clients=10000, clock=monotonic):
        if min(uploads, requests, window, max_clients) <= 0:
            raise ValueError('Rate limits must be positive')
        self.app = app
        self.uploads, self.requests = uploads, requests
        self.window, self.max_clients, self.clock = window, max_clients, clock
        self.buckets = OrderedDict()
        self.lock = Lock()

    def check(self, key, limit):
        now = self.clock()
        with self.lock:
            while self.buckets and next(iter(self.buckets.values()))[0] <= now:
                self.buckets.popitem(last=False)
            if key not in self.buckets:
                if len(self.buckets) >= self.max_clients:
                    return max(1, ceil(next(iter(self.buckets.values()))[0] - now))
                self.buckets[key] = [now + self.window, 0]
            bucket = self.buckets[key]
            if bucket[1] >= limit:
                return max(1, ceil(bucket[0] - now))
            bucket[1] += 1
        return 0

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if (scope['type'] != 'http' or not path.startswith('/api/')
                or path == '/api/health' or scope['method'] == 'OPTIONS'):
            return await self.app(scope, receive, send)
        upload = scope['method'] == 'POST' and path.rstrip('/') == '/api/jobs'
        # Use the ASGI peer address, never client-supplied forwarding headers.
        peer = scope.get('client')
        key = (peer[0] if peer else 'unknown', 'upload' if upload else 'request')
        retry = self.check(key, self.uploads if upload else self.requests)
        if retry:
            response = JSONResponse(
                {'detail': 'Too many requests. Try again later.', 'code': 'rate_limited'},
                status_code=429, headers={'Retry-After': str(retry), 'Cache-Control': 'no-store'})
            return await response(scope, receive, send)
        await self.app(scope, receive, send)
