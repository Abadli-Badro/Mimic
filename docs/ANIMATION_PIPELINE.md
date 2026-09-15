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
  parent's coordinates. Unreliable joints retain their previous local rotations.
- BVH motion uses exactly the recursive hierarchy declaration order.
- GLB includes a canonical joint tree with offsets. It is a skeleton-only file.
- Merging onto a rig transfers source world rotation deltas to target world rest
  frames, then computes target-local rotations. Merely multiplying a local delta
  by a target rest quaternion is insufficient.
- `--fps` resamples rotations using spherical interpolation. It does not simply
  change playback speed. `--format bvh` selects the BVH writer.

## Validation

Run `.venv\Scripts\python.exe -m pytest -q` on Windows. Regression tests cover
identity T-pose, rigid body turns, occlusion, BVH channel round trips, retained
GLB parents, target bind frames, trimming, and frame-rate conversion.

The reference video was rebuilt into `output/fixed/`: 211 frames after
trimming source frames 0-39 and 251-303. `pipeline_check.png` compares source
frames, reconstructed joint positions and exported model joints.
`mesh_check.png` renders the exported mesh using its skin weights and inverse
bind matrices. Intermediate NPZ files are retained there for inspection.

## Limits

These exports are in-place: camera-relative travel and foot-contact locking are
not reconstructed from hip-centered pose landmarks. Single-view limb directions
do not determine full anatomical twist; fingers remain in the model's rest pose.
Spinal articulation is approximated from hips and shoulders. The merge accepts
rotation-only, dense LINEAR tracks on one timeline and TRS target nodes; it rejects
unsupported matrix nodes and source translation/scale channels explicitly.
