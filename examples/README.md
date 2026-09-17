# Example animations

`sample_videos/` contains the two input clips. `animations/` contains each
exported character GLB, full pose overlay, and side-by-side comparison MP4
with a JPG preview. Download links and previews are in the root README.

## Comparison timing

| Clip | Source frames shown (zero-based, inclusive) | Animation frames |
| --- | --- | --- |
| Dance Reference | 0-261 | 262 |
| Sneaky walk reference | 40-234 | 195 |

Comparisons end with the exported animation. The standalone overlays continue
through the full source footage. Both exports use a 20-frame missing-bone limit.

## Regenerate comparisons

First run the two example pipeline commands in the root README. Keep their
`output/intermediate/<clip>/rotations.npz` files: the renderer reads the source
frame offset, frame count, and frame rate from those files.

```cmd
.venv\Scripts\python.exe examples/render_comparisons.py
```

The renderer evaluates the GLB animation tracks, node transforms, inverse bind
matrices, and vertex skin weights. It uses a fixed orthographic front camera and
neutral shaded materials, without textures or audio. Triangle depth sorting is
used for this lightweight preview; a 3D application's rendering may differ.
Dependencies: NumPy, SciPy, pygltflib, and OpenCV.
