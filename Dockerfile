# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mimic

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY mimic/ ./mimic/
COPY interfaces/ ./interfaces/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -e '.[api]'

COPY data/models/pose_landmarker.task data/models/model.glb ./data/models/
RUN mkdir -p /app/output && chown mimic:mimic /app/output

USER mimic
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).close()"

CMD ["python", "-m", "uvicorn", "interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
