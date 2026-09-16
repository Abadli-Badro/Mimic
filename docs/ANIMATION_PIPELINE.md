# Animation pipeline conventions

The canonical skeleton is Y-up and +Z-forward, with anatomical left on +X.
Joint offsets are in meters; BVH converts these to centimeters. Local rest
rotations are identity. `JOINT_OFFSETS` in `retargeting/skeleton.py` is shared
by both exporters and the forward-kinematics preview.

Legacy leg names are retained for compatibility: `left_hip` is the thigh,
`left_upper_leg` is the shin, `left_lower_leg` is the foot, and `left_foot`
is the toes (and likewise on the right).

## Processing

- Extraction trims leading/trailing frames without a detected person and records
  `first_frame`. An entirely undetected clip produces an error.
- Smoothing interpolates low-confidence gaps before filtering; original confidence
  remains available to the solver. Clip offsets survive the intermediate files.
- The solver measures torso/head frames and solves limb directions in the animated
  parent's coordinates. Short unobserved gaps are interpolated within per-bone frame budgets.
- BVH motion uses exactly the recursive hierarchy declaration order.
- The skeleton writer includes a canonical joint tree with offsets. `convert`
  exports that skeleton; `run --format glb` additionally merges the character.
- Merging onto a rig transfers source world rotation deltas to target world rest
  frames, then computes target-local rotations. Merely multiplying a local delta
  by a target rest quaternion is insufficient.
- `--fps` resamples rotations using spherical interpolation. It does not simply
  change playback speed. `--format bvh` selects the BVH writer.

## Validation

Run `.venv\Scripts\python.exe -m pytest -q` on Windows. Regression tests cover
identity T-pose, rigid body turns, occlusion, BVH channel round trips, retained
GLB parents, target bind frames, trimming, and frame-rate conversion.

Additional tests cover malformed video/NPZ/GLB input, duplicate pose suppression,
per-bone timer resets and cutoffs, atomic writes, and full-command orchestration.
Historical previews under `output/fixed/`, `output/Dance_corrected/`, or
`output/Dance_timers/` are snapshots of earlier runs, not guaranteed current output.

## Artifacts and run state

`pipeline.run()` writes checkpoints to `output/intermediate/<video stem>/`.
`landmarks.npz` contains the trimmed raw detections; `landmarks_smooth.npz`
contains the timer-limited, filtered sequence. `rotations.npz` contains `rot_<bone>`
quaternion arrays and metadata including `fps`, `first_frame`, `num_frames`,
`input_frames`, `bone_names`, and `stopped_bones`.

For character GLB output, `skeleton.glb` is retained as the input to the merge.
`status.json` starts as `running`, then becomes `complete`, `failed`, or
`cancelled`. Failure records include the active stage and error. A successful
bone timeout is a shortened clip; it does not mark the run as failed.
The `exported_frames` status field currently records the solved frame count
before optional FPS resampling; final file key counts can differ.

NPZ loading disables pickle and limits expanded data to 512 MiB. Exporters
validate tracks and structures before publishing through atomic file replacement.
JSON glTF embeds its animation buffer so publication remains a single-file write.
Previously completed files can remain after later-stage failures. The run report
is the source of current-run status; output existence alone is insufficient.

An optional overlay uses raw detections and source timing, so it spans the full
video even when the animation ends earlier. `first_frame` aligns trimmed
landmarks with the original source frames.

## Limits

These exports are in-place: camera-relative travel and foot-contact locking are
not reconstructed from hip-centered pose landmarks. Single-view limb directions
do not determine full anatomical twist; fingers remain in the model's rest pose.
Spinal articulation is approximated from hips and shoulders. The merge accepts
rotation-only, dense LINEAR tracks on one timeline and TRS target nodes; it rejects
unsupported matrix nodes and source translation/scale channels explicitly.

## Multiple-person detection

MediaPipe can return two high-confidence proposals for the same person. Pose
selection groups proposals whose corresponding torso landmarks are closer than
35% of the larger torso length (using aspect-corrected image coordinates).
A candidate needs at least three reliable torso landmarks. The multiple-person
error requires two distinct candidates for 0.3 seconds, with a minimum of three
consecutive frames. The primary subject is selected by proximity to the previous
pose rather than detector list order.

This is a heuristic: very close or heavily occluded people can still be missed,
and brief second-person appearances below the confirmation interval are allowed.

## Per-bone missing-frame budgets

A bone is observed only when all its required landmarks meet the visibility
threshold and its measured direction/frame is nondegenerate. Every observation
resets that bone's counter. With a budget of 20, missing frames 1 through 20 are
allowed; missing frame 21 is excluded and ends the whole animation. Timers begin
at the retained clip start, after extraction trims undetected edges.

The cutoff is applied before gap filling and smoothing so observations beyond
an expired timer cannot influence the exported segment. Local rotations between
observations use quaternion spherical interpolation. Parents are interpolated
before their children are solved. Leading/trailing gaps within the budget use
the nearest available rotation; if a bone has never been observed in the retained
segment it uses its rest rotation until its timer expires. Boundary holds are
not extrapolation or a claim of recovered motion.

`input_frames`, `num_frames`, and `stopped_bones` in the saved rotation NPZ record
truncation. `first_frame` still identifies the source-video offset. Bone budgets
can be overridden in config or via the Python API's `bone_limits` mapping.
