"""Primary glTF/GLB export via pygltflib (pure Python, no external engine)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib

from mimic.retargeting.retarget import INTERNAL_TO_MIXAMO, MIXAMO_HIERARCHY
from mimic.retargeting.skeleton import JOINT_OFFSETS


def _collect_bone_order(hierarchy: dict) -> list[str]:
    """Collect bones in parent-first order."""
    order = []
    queue = [name for name, parent in hierarchy.items() if parent is None]
    while queue:
        bone = queue.pop(0)
        order.append(bone)
        for child, parent in hierarchy.items():
            if parent == bone and child not in order:
                queue.append(child)
    return order


def write_gltf(
    rotations: dict[str, np.ndarray],
    fps: float,
    output_path: Path,
    glb: bool = True,
    bone_names: list[str] | None = None,
    hierarchy: dict | None = None,
) -> None:
    """Write bone rotations to a glTF or GLB file.

    Creates a minimal glTF with:
    - A node tree matching the Mixamo bone hierarchy
    - Sampler-based animation with rotation keyframes

    Args:
        rotations: Dict mapping Mixamo bone name -> (num_frames, 4) quaternions (x,y,z,w).
        fps: Frame rate.
        output_path: Path to write the .gltf or .glb file.
        glb: If True, write binary GLB format.
        bone_names: List of bone names in the animation.
        hierarchy: Dict mapping bone -> parent bone.
    """
    if hierarchy is None:
        hierarchy = MIXAMO_HIERARCHY
    if bone_names is None:
        bone_names = list(rotations.keys())

    # Determine frame count
    num_frames = 0
    for v in rotations.values():
        num_frames = v.shape[0]
        break

    if not np.isfinite(fps) or fps <= 0 or num_frames < 1:
        raise ValueError("Expected positive fps and at least one animation frame")
    for quats in rotations.values():
        if np.shape(quats) != (num_frames, 4) or not np.isfinite(quats).all():
            raise ValueError("Expected finite quaternion tracks with equal frame counts")
        if np.any(np.linalg.norm(quats, axis=1) < 1e-8):
            raise ValueError("Zero quaternion is not a rotation")
    rotations = {name: q / np.linalg.norm(q, axis=1, keepdims=True)
                 for name, q in rotations.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf = pygltflib.GLTF2()
    scene = pygltflib.Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # Build bone order (parent-first)
    bone_order = _collect_bone_order(hierarchy)
    # Filter to only bones we have rotations for
    if set(rotations) - set(bone_order):
        raise ValueError("Animation contains bones outside the hierarchy")
    # Keep unanimated parents so descendants retain their rest transforms.
    for bone in bone_order:
        rotations.setdefault(bone, np.tile([0., 0., 0., 1.], (num_frames, 1)))
    offsets = {INTERNAL_TO_MIXAMO[name]: value for name, value in JOINT_OFFSETS.items()}

    # Create nodes for each bone
    node_map: dict[str, int] = {}
    for i, bone_name in enumerate(bone_order):
        offset = offsets.get(bone_name, JOINT_OFFSETS.get(bone_name, (0, 0, 0)))
        node = pygltflib.Node(name=bone_name, translation=list(offset))
        parent = hierarchy.get(bone_name)
        if parent in node_map:
            parent_node = gltf.nodes[node_map[parent]]
            if parent_node.children is None:
                parent_node.children = []
            parent_node.children.append(i)
        gltf.nodes.append(node)
        node_map[bone_name] = i

    scene.nodes = [node_map[b] for b in bone_order if hierarchy[b] is None]

    # Build animation data
    # Sampler input: timestamps
    timestamps = np.arange(num_frames, dtype=np.float32) / fps
    # Sampler output: quaternion keyframes per bone
    all_keyframes = []
    for bone_name in bone_order:
        quats = rotations[bone_name]  # (num_frames, 4)
        all_keyframes.append(quats)
    all_keyframes = np.concatenate(all_keyframes, axis=0).astype(np.float32)

    # Create buffer with timestamps + keyframes
    timestamp_bytes = timestamps.tobytes()
    keyframe_bytes = all_keyframes.tobytes()

    timestamp_buffer_view = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(timestamp_bytes),
        )
    )
    keyframe_buffer_view = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=len(timestamp_bytes),
            byteLength=len(keyframe_bytes),
        )
    )

    # Accessors
    timestamp_accessor = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=timestamp_buffer_view,
            componentType=pygltflib.FLOAT,
            count=len(timestamps),
            type="SCALAR",
            max=[float(timestamps[-1])],
            min=[0.0],
        )
    )

    keyframe_offset = 0
    keyframe_accessors = []
    for bone_name in bone_order:
        quats = rotations[bone_name]
        accessor = len(gltf.accessors)
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=keyframe_buffer_view,
                componentType=pygltflib.FLOAT,
                count=num_frames,
                type="VEC4",
                byteOffset=keyframe_offset,
            )
        )
        keyframe_accessors.append(accessor)
        keyframe_offset += num_frames * 4 * 4  # 4 floats * 4 bytes

    # Animation samplers and channels
    animation = pygltflib.Animation(name="mixamo_animation")
    for i, bone_name in enumerate(bone_order):
        sampler_idx = len(animation.samplers)
        animation.samplers.append(
            pygltflib.AnimationSampler(
                input=timestamp_accessor,
                output=keyframe_accessors[i],
                interpolation="LINEAR",
            )
        )
        animation.channels.append(
            pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(
                    node=node_map[bone_name],
                    path="rotation",
                ),
            )
        )
    gltf.animations.append(animation)

    # Buffer
    combined = timestamp_bytes + keyframe_bytes
    gltf.buffers.append(
        pygltflib.Buffer(byteLength=len(combined))
    )

    # Set binary data
    gltf.set_binary_blob(combined)

    # Save
    if glb:
        gltf.save_binary(str(output_path))
    else:
        gltf.save(str(output_path))
