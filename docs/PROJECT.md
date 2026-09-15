# Feasibility Study: MP4 → MediaPipe → Mixamo-Rigged glTF Animation

**Scope:** Python pipeline. Input: an MP4 video of a person moving. Output: a glTF/GLB animation clip, using Mixamo's standard skeleton, applied to a default 3D character. (Updated from the original FBX target — see §6 for why.)

**Verdict up front:** Feasible as a personal project, with real technical difficulty concentrated in one specific stage — converting sparse 3D joint _positions_ into bone _rotations_ that a rig can actually use. Everything else (extraction, export) is well-supported by existing tools, and with glTF as the target, the whole pipeline can run in pure Python with no external engine dependency. Budget for a "good enough, visibly imperfect" MVP before a polished one.

---

## 1. Pipeline stages and feasibility

| Stage                                          | Tool                               | Feasibility                                                                                                                  |
| ---------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Read MP4, extract frames                       | OpenCV                             | Trivial                                                                                                                      |
| Detect 3D body landmarks per frame             | MediaPipe Pose Landmarker          | Easy to get running; quality is the limiting factor                                                                          |
| Convert landmark trajectories → bone rotations | Custom math (NumPy/SciPy)          | **This is the hard part of the whole project**                                                                               |
| Smooth / clean the motion                      | SciPy / One-Euro filter            | Easy, well-documented technique                                                                                              |
| Retarget onto Mixamo skeleton                  | Custom mapping table (pure Python) | Moderate — mostly bookkeeping, some fiddly edge cases                                                                        |
| Bake and export as glTF/GLB                    | `pygltflib` (pure Python)          | Well-supported — glTF is an open, documented format built around exactly this kind of skinned-mesh + keyframe animation data |

MediaPipe Pose Landmarker is actively maintained and outputs both normalized 2D-ish landmarks and **world landmarks in meters**, giving you a genuine (if noisy) 3D skeleton per frame — that part of the stack is solid and current.

glTF/GLB is an open, well-documented format with a pure-Python writer (`pygltflib`) — no Blender, no subprocess, no external engine at all. It's also the native format for web 3D (three.js / react-three-fiber load it directly), which matters given the eventual FastAPI + React web app. FBX is still possible as an optional secondary export later (via `pip install bpy`, which runs Blender in-process without a subprocess) for people who specifically need it in Unity/Unreal, but it's no longer the core dependency.

---

## 2. Where the real difficulty lives

**2.1 MediaPipe gives you points, not bones.**
Pose Landmarker outputs 33 XYZ positions per frame (joints like shoulder, elbow, wrist). A rig, however, animates via bone _rotations_ (quaternions), not point positions. You have to derive each bone's rotation yourself: for every frame, build a direction vector between a joint and its child (e.g., shoulder → elbow), compare it to that bone's rest-pose direction, and solve for the quaternion that rotates one into the other. This is standard but non-trivial vector math, and it's the crux of the entire project — get this wrong and the character looks broken even if the raw tracking looked fine.

**2.2 Monocular depth ambiguity.**
A single MP4 has no true depth information — MediaPipe's "world landmarks" are inferred, not measured, and get noisier and less reliable for motion toward/away from the camera, for occluded limbs, and for fast motion (motion blur). Don't expect studio-mocap-grade cleanliness. Side-on or awkward-angle poses (e.g., a hand crossing behind the body) are the failure cases you'll hit repeatedly.

**2.3 No twist/roll information.**
Joint positions alone can't tell you things like forearm roll (pronation/supination) or spine twist — you only see where the _ends_ of a bone are, not how it's rotated around its own axis. You'll need heuristics or just accept some bones (mainly forearms) won't animate that degree of freedom.

**2.4 No root translation / world-scale motion.**
MediaPipe's world landmarks are centered on the hips, not anchored to the real world — so if the person walks across the room, that root motion isn't given to you directly. If the project needs the character to _move through space_ (not just animate in place), you'll need an extra heuristic (e.g., tracking hip position drift in the 2D frame + scale estimation) — this is a known "hard mode" add-on, not core MVP.

**2.5 Skeleton mismatch.**
MediaPipe's 33-point skeleton has no fingers, no spine subdivisions, and a simplified structure. Mixamo's rig has ~65 bones including individual finger joints and multiple spine/neck bones. Mapping is one-to-many and incomplete: some Mixamo bones (fingers, toes, some spine segments) simply have nothing to drive them and will need to stay static or be interpolated from neighboring bones.

**2.6 Coordinate system / axis convention hell.**
MediaPipe and glTF use different up-axes and handedness conventions (glTF is +Y-up, right-handed, which is at least a fixed, well-documented target — no per-app axis quirks like Blender/Unity/Unreal add). This is still a very common, fiddly source of "why is my character's leg twisted 90°" bugs. Budget real debugging time here — it's tedious rather than conceptually hard.

**2.7 Rest pose mismatch (A-pose vs T-pose).**
Mixamo characters are typically rigged in an A-pose; your rotation math needs to account for the rig's actual rest pose, not assume a T-pose, or every bone will be offset by a constant rotation error.

None of these are blockers — they're all solved problems in the broader mocap-retargeting space — but they add up to genuine engineering work, not a weekend script.

---

## 3. Suggested scope

**MVP (realistic first milestone):**

- Single person, roughly front-facing, single stationary camera
- Body only — no fingers, no facial animation
- In-place motion only (no root translation — character animates but doesn't travel)
- One output clip per input video
- Visible jitter/imperfection acceptable
- Output: glTF/GLB with baked animation on a Mixamo-compatible skeleton, importable into Blender/Unity/Unreal and loadable directly in a browser

**Stretch goals (post-MVP):**

- Hand/finger tracking (needs MediaPipe Holistic or a separate Hand Landmarker pass)
- Root motion / world-space translation
- Foot IK / ground-contact cleanup (prevents foot sliding — a classic mocap-retargeting artifact)
- Multiple people in frame
- Batch processing / a small library of default actions

---

## 4. Step-by-step plan

**— Extraction**
Read MP4 with OpenCV, run MediaPipe Pose Landmarker per frame, dump the 33-landmark world-coordinate sequence to a `.npz`/`.json` file. Visualize a few frames with matplotlib as a 3D scatter to sanity-check tracking quality before building anything else.

**— Cleaning**
Apply a temporal filter (One-Euro or Savitzky-Golay) per landmark across frames to remove jitter. Plot before/after to confirm it's actually helping and not over-smoothing fast motion.

**— Rotation solving (core engineering)**
Define a simple skeleton hierarchy matching MediaPipe's landmark set (parent/child pairs). For each bone, compute per-frame rotation relative to a rest pose using vector alignment (e.g., via `scipy.spatial.transform.Rotation.align_vectors`). Validate by applying the rotations to a simple stick-figure skeleton and animating it — no rig or FBX yet, just confirm the math is sound.

**— BVH export**
Write the rotation data out as a BVH file (simple, human-readable, universal mocap format). This is a good checkpoint: BVH import into Blender is standard, letting you visually verify the motion before tackling retargeting complexity.

**— Retargeting to Mixamo skeleton**
Build a name/hierarchy mapping table from your simplified skeleton to Mixamo's bone names, and apply your BVH-derived rotations onto the target skeleton's joint hierarchy in pure Python, accounting for the rest-pose offset (A-pose correction).

**— glTF export**
Write the retargeted skeleton, its skinning data, and per-frame rotation keyframes to a `.glb` using `pygltflib`. Test the result by loading it in a quick three.js/react-three-fiber viewer (or Blender's glTF importer) to confirm it plays correctly outside your working environment.

**— CLI + polish**
Wrap the above stages behind a `typer`-based CLI (`extract`, `clean`, `retarget`, `export`), with intermediate artifacts saved at each step for debuggability. Add axis-convention fixes and pose-mismatch corrections as you find them through testing.

---

## 5. Recommended stack

- **Extraction:** `opencv-python`, `mediapipe`
- **Math/filtering:** `numpy`, `scipy` (rotation utilities, Savitzky-Golay)
- **Rigging/export:** `pygltflib` (pure Python, no external engine)
- **CLI:** `typer` or `click`
- **Intermediate format:** BVH (for debugging/visual checks) → glTF/GLB (final output)

This keeps everything free, scriptable, in-process, and dependency-light — no Blender, no subprocess management, no unmaintained Autodesk FBX SDK.
