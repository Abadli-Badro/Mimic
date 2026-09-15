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

## CLI help

```bash
mimic --help
mimic convert --help
mimic merge -h
```

Running `mimic` without arguments also displays help. Each command supports
`-h` and `--help`, including examples and default output paths.

| Command | Purpose | Example |
| --- | --- | --- |
| `extract` | Save pose landmarks to NPZ | `mimic extract video.mp4 --visualize` |
| `smooth` | Filter extracted landmarks | `mimic smooth output/video.npz` |
| `convert` | Export a skeleton animation | `mimic convert video.mp4 --format bvh` |
| `visualize` | Overlay landmarks on the source video | `mimic visualize video.mp4` |
| `merge` | Animate a rigged GLB character | `mimic merge data/models/model.glb output/video.glb` |
| `info` | Show the installed version | `mimic info` |

Use `convert --fps 30` to resample animation. GLB conversion produces a skeleton;
use `merge` to add the rigged character. `visualize` uses the NPZ from `extract`
by default; use `--npz output/intermediate/video/landmarks.npz` after `convert`.

## Project Structure

- `mimic/` - Core library (extraction, processing, retargeting, export)
- `interfaces/cli/` - Typer CLI (thin wrapper around core)
- `interfaces/api/` - Future FastAPI backend (placeholder)
- `data/` - Models and rigs
- `tests/` - Unit tests

See [the module guide](docs/STRUCTURE.md) and
[animation conventions](docs/ANIMATION_PIPELINE.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
