# Project structure

Mimic separates file workflows from numerical algorithms. The CLI calls the
library; `mimic.pipeline.run()` orchestrates conversion from video to animation.

```text
mimic/
    pipeline.py                 Full conversion workflow
    config.py                   Paths, resource limits, and tracking defaults
    errors.py                   Stage-aware errors and stable error codes
    validation.py               Landmark, hierarchy, and rotation contracts
    artifacts.py                Safe NPZ loading, atomic writes, and run status
    extraction/
        video_reader.py         Video decoding and frame sampling
        pose_extractor.py       MediaPipe landmark detection
        pose_selection.py       Duplicate suppression and primary pose selection
        quality.py              Human-pose usability checks
        video_to_landmarks.py   Extract and save a landmark NPZ file
        overlay.py              Draw landmarks over the original video
        landmark_plots.py       Plot extracted 3D landmarks
    processing/
        smoothing.py            Temporal filters and gap interpolation
        smooth_landmarks.py     Load, smooth, and save a landmark NPZ file
        rotation_solver.py      Compute/interpolate local bone rotations
        tracking.py             Per-bone timers and common clip cutoff
        landmarks_to_rotations.py  Load landmarks and save rotation tracks
        trajectory_plots.py     Compare raw and filtered trajectories
        skeleton_preview.py     Landmark previews and forward kinematics
    retargeting/
        skeleton.py             Canonical skeleton and rest offsets
        retarget.py             Internal-to-Mixamo names and hierarchy
        mixamo_mapping.json     Legacy mapping reference
    export/
        bvh_writer.py           BVH hierarchy and animation serialization
        gltf_writer.py          Skeleton GLB/glTF serialization
        merge_animation.py      Transfer animation onto a skinned model
        gltf_validation.py      Container, graph, skin, and animation checks
        fbx_exporter.py         Unimplemented optional FBX exporter
interfaces/
    cli/main.py                 Typer commands
    api/                        Reserved API package
tests/
    test_video_to_landmarks.py  Video-to-NPZ workflow checks
    test_extraction.py          Video reader, detector, and preview checks
    test_processing.py          Filter and solver checks
    test_retargeting.py         Skeleton and mapping checks
    test_export.py              Export checks
    test_pipeline.py            Pipeline checks
    test_animation_regressions.py  Motion and export regression checks
    test_error_handling.py      Invalid inputs and failure-path checks
    test_pose_selection.py      Duplicate and multi-person detection checks
    test_tracking_timers.py     Gap interpolation and timeout checks
    test_run_command.py         Full-pipeline command orchestration
data/
    input/                      Source clips and existing reference artifacts
    models/                     MediaPipe model and rigged character assets
output/                         Generated animations and diagnostic artifacts
    intermediate/               Retained extraction, smoothing, and rotation files
    previews/                   Landmark plots
docs/
    STRUCTURE.md                This module guide
    ANIMATION_PIPELINE.md        Coordinate conventions and validation
    PROJECT.md                  Current scope and supported behavior
```

Each package also contains an `__init__.py` file.

## Naming and responsibilities

- Name modules after their responsibility, not a numbered development milestone.
- Keep numerical operations in `smoothing.py` and `rotation_solver.py`; file
  workflows handle NPZ loading, metadata, and saving.
- Keep visualization separate from extraction and solving. Use
  `skeleton_preview.reconstruct_joints()` to inspect solved rotations.
- Keep argument parsing in `interfaces/cli/main.py` and reusable work in `mimic/`.
- Keep generated assets in `output/` rather than beside source code.

## Entry points

The installed `mimic` command points to `interfaces.cli.main:app`. It exposes
`run`, `extract`, `smooth`, `convert`, `visualize`, `merge`, and `info`.

Library workflows can also be called directly:

```python
from mimic.extraction.video_to_landmarks import extract
from mimic.processing.smooth_landmarks import smooth
from mimic.processing.landmarks_to_rotations import solve
from mimic.pipeline import run
```

See [animation conventions](ANIMATION_PIPELINE.md) for coordinate systems,
retargeting behavior, and current limitations.
