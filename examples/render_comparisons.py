"""Render example GLBs beside their overlays using NumPy and OpenCV.

Run from the project root: python examples/render_comparisons.py
Requires the matching output/intermediate/<name>/rotations.npz for source timing.
The preview uses neutral shaded materials and an orthographic front camera.
"""
from pathlib import Path

import cv2
import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def accessor(gltf, index):
    a = gltf.accessors[index]
    view = gltf.bufferViews[a.bufferView]
    dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a.componentType])
    size = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a.type]
    return np.ndarray((a.count, size), dtype=dtype, buffer=gltf.binary_blob(),
                      offset=(view.byteOffset or 0) + (a.byteOffset or 0),
                      strides=(view.byteStride or size * dtype.itemsize, dtype.itemsize)).copy()


def load_scene(path):
    g = GLTF2.load(str(path))
    meshes = []
    for n in g.nodes:
        if n.mesh is None:
            continue
        skin = g.skins[n.skin]
        inverse = accessor(g, skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
        for p in g.meshes[n.mesh].primitives:
            positions = accessor(g, p.attributes.POSITION)
            meshes.append((np.column_stack([positions, np.ones(len(positions))]),
                           accessor(g, p.attributes.JOINTS_0).astype(int),
                           accessor(g, p.attributes.WEIGHTS_0),
                           accessor(g, p.indices).reshape(-1, 3), skin.joints, inverse))
    tracks = []
    for c in g.animations[0].channels:
        s = g.animations[0].samplers[c.sampler]
        assert s.interpolation in ('LINEAR', None), 'Unsupported interpolation'
        times = accessor(g, s.input).ravel()
        values = accessor(g, s.output)
        tracks.append((c.target.node, c.target.path, times, values))
    return g, meshes, tracks


def deform(scene, time):
    g, meshes, tracks = scene
    translations = [np.array(n.translation or [0, 0, 0], dtype=float) for n in g.nodes]
    rotations = [np.array(n.rotation or [0, 0, 0, 1], dtype=float) for n in g.nodes]
    scales = [np.array(n.scale or [1, 1, 1], dtype=float) for n in g.nodes]
    for node, path, times, values in tracks:
        t = np.clip(time, times[0], times[-1])
        if path == 'rotation':
            rotations[node] = Slerp(times, Rotation.from_quat(values))([t]).as_quat()[0]
        else:
            value = np.array([np.interp(t, times, values[:, i]) for i in range(3)])
            {'translation': translations, 'scale': scales}[path][node] = value
    local = []
    for i, n in enumerate(g.nodes):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(rotations[i]).as_matrix() @ np.diag(scales[i])
        m[:3, 3] = translations[i]
        local.append(np.array(n.matrix).reshape(4, 4).T if n.matrix else m)
    parents = {child: i for i, n in enumerate(g.nodes) for child in n.children}
    world = {}

    def matrix(i):
        if i not in world:
            world[i] = matrix(parents[i]) @ local[i] if i in parents else local[i]
        return world[i]

    result = []
    for pos, joints, weights, faces, skin_joints, inverse in meshes:
        matrices = np.array([matrix(i) for i in skin_joints]) @ inverse
        vertices = np.zeros((len(pos), 4))
        for k in range(4):
            vertices += np.einsum('nij,nj->ni', matrices[joints[:, k]], pos) * weights[:, k, None]
        result.append((vertices[:, :3], faces))
    return result


def render(name):
    folder = ROOT / 'examples' / 'animations'
    scene = load_scene(folder / f'{name}.glb')
    with np.load(ROOT / 'output' / 'intermediate' / name / 'rotations.npz') as meta:
        start, count, fps = int(meta['first_frame']), int(meta['num_frames']), float(meta['fps'])
    # Fit one fixed camera to the entire clip so the model never changes size.
    poses = [deform(scene, i / fps) for i in range(count)]
    all_vertices = np.concatenate([v for pose in poses for v, _ in pose])
    low, high = all_vertices.min(axis=0), all_vertices.max(axis=0)
    center = (low + high) / 2
    scale = min(560 / (high[0] - low[0]), 430 / (high[1] - low[1]))
    cap = cv2.VideoCapture(str(folder / f'{name}_overlay.mp4'))
    assert cap.isOpened()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    destination = folder / f'{name}_comparison.mp4'
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 560))
    assert writer.isOpened()
    try:
        for i, pose in enumerate(poses):
            ok, frame = cap.read()
            assert ok, f'Missing overlay frame {start+i}'
            canvas = np.full((560, 1280, 3), (26, 22, 19), np.uint8)
            h, w = frame.shape[:2]
            factor = min(620 / w, 450 / h)
            frame = cv2.resize(frame, (round(w * factor), round(h * factor)))
            h, w = frame.shape[:2]
            x, y = (640 - w) // 2, 60 + (450 - h) // 2
            canvas[y:y+h, x:x+w] = frame
            triangles, colors, depths = [], [], []
            for part, (vertices, faces) in enumerate(pose):
                projected = np.column_stack([960 + (vertices[:, 0] - center[0]) * scale,
                                             285 - (vertices[:, 1] - center[1]) * scale])
                tri = vertices[faces]
                normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
                light = 0.35 + 0.65 * np.abs(normals @ np.array([0.3, 0.5, 0.8124]))
                base = np.array([210, 177, 99] if part == 0 else [115, 105, 85])
                triangles.extend(np.rint(projected[faces]).astype(np.int32))
                colors.extend(np.clip(light[:, None] * base, 0, 255).astype(np.uint8))
                depths.extend(tri[:, :, 2].mean(axis=1))
            for index in np.argsort(depths):
                cv2.fillConvexPoly(canvas, triangles[index], tuple(int(v) for v in colors[index]), cv2.LINE_AA)
            cv2.line(canvas, (640, 15), (640, 535), (70, 62, 55), 1)
            for title, x in [('SOURCE + POSE OVERLAY', 24), ('EXPORTED GLB / SHADED MESH', 665)]:
                cv2.putText(canvas, title, (x, 35), cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 235, 235), 1, cv2.LINE_AA)
            label = f'{name}  |  source frame {start+i}  |  {(start+i)/fps:.2f}s'
            cv2.putText(canvas, label, (24, 540), cv2.FONT_HERSHEY_SIMPLEX, .55, (190, 190, 190), 1, cv2.LINE_AA)
            writer.write(canvas)
            if i == count // 2:
                cv2.imwrite(str(folder / f'{name}_comparison.jpg'), canvas)
    finally:
        cap.release()
        writer.release()
    print(f'{destination.name}: {count} frames, source frames {start}-{start+count-1}', flush=True)


if __name__ == '__main__':
    for name in ('Dance Reference', 'Sneaky walk reference'):
        render(name)
