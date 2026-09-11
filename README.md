# Mimic

Convert MP4 video of a person moving into a glTF/GLB 3D animation using a Mixamo-compatible skeleton.

## Installation

```bash
pip install -e .
```

## Usage

```bash
mimic convert video.mp4
mimic convert video.mp4 -o output.glb --fps 30
mimic info
```

## Project Structure

- `mimic/` — Core library (extraction, processing, retargeting, export)
- `interfaces/cli/` — Typer CLI (thin wrapper around core)
- `interfaces/api/` — Future FastAPI backend (placeholder)
- `data/` — Models and rigs
- `tests/` — Unit tests

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
