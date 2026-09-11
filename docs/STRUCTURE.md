# Mimic — Project Structure

Design goal: the CLI is a thin wrapper around a core library. Nothing pipeline-related lives inside the CLI code — so a future FastAPI backend can import the exact same core and expose it over HTTP without touching the pipeline logic at all.

```
mimic/
├── mimic/                          # core library — no CLI or web code lives here
│   ├── __init__.py
│   ├── config.py                   # paths, constants, skeleton definitions
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── video_reader.py         # OpenCV frame iteration
│   │   └── pose_extractor.py       # MediaPipe wrapper -> per-frame landmarks
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── smoothing.py            # temporal filtering (One-Euro / Savitzky-Golay)
│   │   └── rotation_solver.py      # landmark positions -> bone rotations
│   │
│   ├── retargeting/
│   │   ├── __init__.py
│   │   ├── skeleton.py             # internal skeleton hierarchy definition
│   │   ├── mixamo_mapping.json     # internal bone name -> Mixamo bone name
│   │   └── retarget.py             # applies rotations onto target rig
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   ├── bvh_writer.py           # intermediate BVH output (debug checkpoint)
│   │   ├── gltf_writer.py          # PRIMARY output — pure Python via pygltflib, no external engine
│   │   └── fbx_exporter.py         # OPTIONAL secondary export, via bpy (see below)
│   │
│   └── pipeline.py                 # orchestrates the full flow end-to-end;
│                                    # THE single entrypoint both CLI and future API call
│
├── interfaces/
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                 # typer app — thin, just parses args and calls mimic.pipeline
│   │
│   └── api/                        # placeholder — future FastAPI backend
│       └── (not built yet: would import mimic.pipeline directly, add routers/schemas here)
│
├── web/                             # placeholder — future React frontend
│   └── (not built yet)
│
├── data/
│   ├── models/                     # downloaded MediaPipe .task model files (gitignored)
│   └── rigs/                       # sample Mixamo character files for local testing
│
├── examples/
│   └── sample_videos/              # small test clips for development
│
├── tests/
│   ├── test_extraction.py
│   ├── test_processing.py
│   ├── test_retargeting.py
│   └── test_pipeline.py
│
├── pyproject.toml                  # single installable package, CLI entrypoint declared here
├── README.md
├── .gitignore
└── LICENSE
```

## Why this shape

- **`mimic/` is the product; `interfaces/` are just doors into it.** The CLI should never contain business logic — it should read like "parse these args, call `pipeline.run(...)`, print the result." That's what makes bolting on `interfaces/api/` later a small change instead of a rewrite.
- **`pipeline.py` is the seam.** One function (or a small set of them) that takes an input video path and options, and returns/writes the final FBX. The CLI calls it synchronously; a future FastAPI endpoint calls the same function, probably wrapped in a background task since these pipelines take real time to run.
- **glTF is the primary export format, not FBX.** `gltf_writer.py` uses `pygltflib` — pure Python, no external engine, no subprocess. This also happens to be the right call for the web app: browsers (three.js / react-three-fiber) load glTF/GLB natively, whereas FBX can't be loaded client-side at all without a server-side conversion step first.
- **`fbx_exporter.py` is optional and isolated on purpose.** If FBX output is ever needed (e.g. for someone who specifically wants to import into Blender/Unity/Unreal), it would use `pip install bpy` (Blender as an in-process Python module — no subprocess, no separate Blender install required) purely inside this one module, so the rest of the codebase — and the whole web path — never depends on it.
- **`web/` stays empty until you actually start it.** No point scaffolding a React app before there's an API to point it at — the folder's there so the eventual repo layout doesn't need restructuring, but there's nothing to maintain until you need it.

## `pyproject.toml` entry point (CLI)

```toml
[project.scripts]
mimic = "interfaces.cli.main:app"
```

This gives you a global `mimic` command (e.g. `mimic extract video.mp4`) once the package is installed with `pip install -e .`.

## When you get to the web app later

- `interfaces/api/` becomes a FastAPI app with routers like `POST /jobs` (upload video, kick off pipeline), `GET /jobs/{id}` (poll status), `GET /jobs/{id}/result` (download the `.glb`) — all calling straight into `mimic.pipeline`.
- Long-running pipeline runs should go through a task queue (even something simple like FastAPI's `BackgroundTasks`, or Celery/RQ if it grows) rather than blocking the HTTP request.
- `web/` becomes a standard Vite + React app that talks to the FastAPI backend over REST — and since output is glTF/GLB, it can preview the result directly in the browser with react-three-fiber, no conversion step needed.
