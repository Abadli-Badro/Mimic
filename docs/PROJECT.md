# Project scope

Mimic is a local Python video-to-animation tool for single-person footage. It
uses OpenCV to decode video, MediaPipe to estimate body landmarks, NumPy/SciPy
to filter motion and solve rotations, and pygltflib to export animation.

## Supported workflows

- `mimic run`: one command for animated-character GLB or skeleton BVH, with
  optional overlay video and per-bone missing-frame limits.
- `mimic convert`: skeleton-only GLB, glTF, or BVH.
- `extract`, `smooth`, `visualize`, and `merge`: inspect or run individual steps.
- Python functions expose the same operations for reuse outside the CLI.

Generated files live under project-root `output/`. Inputs and model assets live
under `data/`. Intermediate files retain source-frame offsets so overlays stay
aligned after trimming. See the [README](../README.md) for setup and commands.

## Processing model

1. Validate video metadata and resource budgets, then decode frames.
2. Detect pose candidates, suppress duplicates, and track the primary person.
   Reject a persistent, credible second person and clips without usable poses.
3. Trim undetected clip edges. Compute each bone's observation mask and end the
   animation when the first missing-frame budget is exceeded.
4. Fill landmark gaps within the retained segment and apply One-Euro filtering.
5. Solve torso/head frames and limb rotations. Interpolate short gaps in local
   quaternion tracks, completing each parent before solving its children.
6. Export the canonical skeleton or transfer its rotation deltas through the
   character's rest hierarchy for GLB. Optionally overlay raw detections on the
   full source video.

## Reliability boundaries

Landmarks are inferred from a single view. They do not provide measured depth,
root travel, complete bone twist, or detailed spinal articulation. Short gaps
can be interpolated, but longer gaps end the clip rather than freezing forever.

Two detected poses do not necessarily mean two people: overlapping proposals
are deduplicated and a second distinct candidate must persist. Conversely, close
or occluded people can be missed. Pose detection is not an animal classifier.

Validation covers video resources, numeric landmark data, rotation tracks,
skeleton graphs, GLB buffers/timelines, and supported target rigs. Generated
files use atomic replacement; run status records completion or the failed stage.

## Current limits and future work

- Motion is in place; root translation and foot-contact cleanup are future work.
- Finger articulation and reliable anatomical limb twist are not reconstructed.
- Multiple-person motion capture and identity selection are not supported.
- FBX export remains a stub; the API package is a placeholder.
- Repeated runs with the same video stem share intermediate paths. Concurrent
  jobs and isolated job storage require additional orchestration.

This document describes the implementation, rather than the original feasibility
plan. See [pipeline details](ANIMATION_PIPELINE.md) and the
[module guide](STRUCTURE.md) for engineering conventions.
