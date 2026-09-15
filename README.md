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

Generated files default to the project-root `output/` directory. Conversion
checkpoints are kept in `output/intermediate/<clip>/`; use `-o` to override a
final output path.

## Project Structure

- `mimic/` â€” Core library (extraction, processing, retargeting, export)
- `interfaces/cli/` â€” Typer CLI (thin wrapper around core)
- `interfaces/api/` â€” Future FastAPI backend (placeholder)
- `data/` â€” Models and rigs
- `tests/` â€” Unit tests

See [the module guide](docs/STRUCTURE.md) and
[animation conventions](docs/ANIMATION_PIPELINE.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
