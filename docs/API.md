# Local conversion API

## Docker

Use Docker Desktop with Linux containers. From the project root:

```cmd
docker compose up --build -d
docker compose logs -f api
```

This builds `mimic-api:latest` and serves `http://localhost:8000/docs`.
The image includes `data/models/pose_landmarker.task` and `data/models/model.glb`;
both must exist before building. It runs as a non-root user and installs the
EGL/GLES libraries required by MediaPipe. Local uploads, archives, examples,
virtual environments, and Git metadata are excluded from the build context.

Outputs persist in the `mimic-output` named volume across container replacement.
Stop with `docker compose down`; adding `-v` also deletes that volume and its jobs.
Keep one API container per output volume. The API retains its 50 MB upload limit.

To build the image without starting it:

```cmd
docker build -t mimic-api:latest .
```

## Python

Install and start from the project root:

```cmd
python -m pip install -e ".[api]"
python -m uvicorn interfaces.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the endpoint schema.
Use one server process, without `--workers` or `--reload` during conversions.
This first version is for local use, without authentication or cross-origin access.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Server liveness |
| POST | `/api/jobs` | Stream a video upload and return a job ID (202) |
| GET | `/api/jobs/{id}` | Status, current stage, timing, errors, download links |
| GET | `/api/jobs/{id}/files/animation` | Download the completed GLB or BVH |
| GET | `/api/jobs/{id}/files/overlay` | Download the optional overlay MP4 |
| DELETE | `/api/jobs/{id}` | Remove a completed or failed job and its outputs |

Upload raw MP4 bytes, **not multipart form data**, with `Content-Type: video/mp4`.
Options are query parameters: `format=glb|bvh`, `overlay=true|false`,
`max_missing_frames=20`, and optional `fps=1..240`. GLB uses
`data/models/model.glb`; pose extraction requires `data/models/pose_landmarker.task`.

```cmd
curl.exe -X POST "http://127.0.0.1:8000/api/jobs?format=glb&overlay=true&max_missing_frames=20" -H "Content-Type: video/mp4" --data-binary "@examples/sample_videos/Sneaky walk reference.mp4"
curl.exe "http://127.0.0.1:8000/api/jobs/JOB_ID"
curl.exe "http://127.0.0.1:8000/api/jobs/JOB_ID/files/animation" -o animation.glb
```

The frontend can send a browser `File` directly as the fetch request body.
Poll status about once per second until `complete` or `failed`.
`progress.first_frame`, `source_frames`, and `source_fps` align video playback
with the animation. `stopped_bones` explains a successful shortened export.

## Resource use and lifecycle

- Upload bytes are streamed to disk; the 50 MB (50,000,000 bytes) limit is enforced even without
  a Content-Length header. Empty uploads return 400; oversized uploads return 413.
- Four slots cover uploading, queued, and running jobs combined. A full queue
  returns 429 with Retry-After. A single spawned process runs conversions so
  MediaPipe work cannot block the HTTP event loop or multiply decoded-video memory.
- Each job has a generated UUID directory under `output/jobs/`. Filenames from
  clients are never used as filesystem paths. Downloads allow only known outputs.
- Pipeline intermediates are deleted on normal exit, and uploads are removed
  after success or failure. Final artifacts remain until explicitly deleted.
- Status is written atomically to disk and survives restart. Interrupted jobs
  are marked failed on startup; they are not automatically retried. Graceful
  shutdown waits for accepted conversions. Abrupt termination may leave scratch
  directories in `output/intermediate/`.
- Separate Uvicorn instances must not share a job root. Public hosting, durable
  distributed queues, authentication, cancellation, and automatic expiry are
  outside this local API's scope. No throughput benchmark is claimed.
