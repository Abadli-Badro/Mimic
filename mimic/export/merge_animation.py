"""Merge skeleton animation onto a full Mixamo rigged model (mesh + skin)."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation


def merge_animation(
    model_path: Path,
    animation_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Take animation keyframes from one GLB and apply them to another GLB's matching nodes.

    Transfers world-space rotation deltas from the canonical source skeleton
    to the target's rest frames, then recovers target-local rotations through
    the hierarchy. Animation channels replace the target local rotation.

    Also strips any existing animations from the model to avoid conflicts.

    Args:
        model_path: Full rigged model .glb (mesh, skin, T-pose skeleton).
        animation_path: Skeleton-only .glb with rotation animation.
        output_path: Output .glb with full mesh + animation.

    Returns:
        Path to output file.
    """
    if output_path is None:
        output_path = output_file(animation_path.stem + "_animated.glb")
    model = pygltflib.GLTF2().load(str(model_path))
    anim_gltf = pygltflib.GLTF2().load(str(animation_path))

    if not anim_gltf.animations:
        raise ValueError(f"No animations in {animation_path}")
    if not model.skins:
        raise ValueError(f"No skin in {model_path}")

    # Strip existing animations — our animation replaces them
    if model.animations:
        print(f"Stripping {len(model.animations)} existing animation(s) from model")
        model.animations.clear()

    # Build name -> node index for both
    model_nodes: dict[str, int] = {}
    for i, node in enumerate(model.nodes):
        if node.name:
            model_nodes[node.name] = i
            model_nodes[node.name.replace("_", ":")] = i

    anim_nodes: dict[str, int] = {}
    for i, node in enumerate(anim_gltf.nodes):
        if node.name:
            anim_nodes[node.name] = i

    source_anim = anim_gltf.animations[0]

    # Find matching bones
    matches: list[tuple[int, int, int]] = []  # (channel_idx, source_sampler_idx, model_node_idx)
    for ch_idx, channel in enumerate(source_anim.channels):
        src_node = anim_gltf.nodes[channel.target.node]
        if channel.target.path == "rotation" and src_node.name in model_nodes:
            matches.append((
                ch_idx,
                channel.sampler,
                model_nodes[src_node.name],
            ))

    print(f"Model nodes: {len(model.nodes)}")
    print(f"Animation channels: {len(source_anim.channels)}")
    print(f"Matched bones: {len(matches)}")

    if not matches:
        raise ValueError("No matching bones found")

    # Read source animation binary blob
    src_blob = anim_gltf.binary_blob()
    if src_blob is None or len(src_blob) == 0:
        raise ValueError("No binary blob in animation GLB")

    # Read timestamps
    ts_sampler_idx = matches[0][1]
    ts_acc_idx = source_anim.samplers[ts_sampler_idx].input
    ts_acc = anim_gltf.accessors[ts_acc_idx]
    ts_bv = anim_gltf.bufferViews[ts_acc.bufferView]
    ts_offset = (ts_bv.byteOffset or 0) + (ts_acc.byteOffset or 0)
    timestamps = np.frombuffer(
        src_blob[ts_offset:ts_offset + ts_acc.count * 4],
        dtype=np.float32,
    )
    num_frames = ts_acc.count
    fps = float(num_frames - 1) / float(timestamps[-1]) if timestamps[-1] > 0 else 30.0

    # Read full source rotations, including unanimated ancestors.
    def read_quats(sampler):
        acc = anim_gltf.accessors[sampler.output]
        bv = anim_gltf.bufferViews[acc.bufferView]
        if (sampler.interpolation not in (None, "LINEAR") or acc.type != "VEC4"
                or acc.componentType != pygltflib.FLOAT or acc.count != num_frames
                or acc.sparse is not None):
            raise ValueError("Merge requires dense LINEAR quaternion tracks with a shared timeline")
        if sampler.input != ts_acc_idx:
            raise ValueError("Merge requires a shared timeline")
        return np.ndarray((acc.count, 4), dtype="<f4", buffer=src_blob,
                          offset=(bv.byteOffset or 0) + (acc.byteOffset or 0),
                          strides=(bv.byteStride or 16, 4)).copy()

    source_tracks = {}
    for channel in source_anim.channels:
        if channel.target.path == "rotation":
            source_tracks[channel.target.node] = Rotation.from_quat(
                read_quats(source_anim.samplers[channel.sampler]))
        else:
            raise ValueError("Merge currently supports rotation-only source animation")

    def local_rest(node):
        if node.matrix:
            raise ValueError("Decompose matrix node transforms to TRS before merging")
        return Rotation.from_quat(node.rotation or [0, 0, 0, 1])

    def parents_and_order(nodes):
        parents = {child: i for i, node in enumerate(nodes) for child in node.children or []}
        order = []
        def visit(i):
            order.append(i)
            for child in nodes[i].children or []:
                visit(child)
        for i in range(len(nodes)):
            if i not in parents:
                visit(i)
        return parents, order

    src_parents, src_order = parents_and_order(anim_gltf.nodes)
    src_rest, src_world = {}, {}
    for i in src_order:
        parent = src_parents.get(i)
        rest = local_rest(anim_gltf.nodes[i])
        src_rest[i] = src_rest.get(parent, Rotation.identity()) * rest
        src_world[i] = src_world.get(parent, Rotation.identity()) * source_tracks.get(i, rest)

    target_parents, target_order = parents_and_order(model.nodes)
    source_for_target = {target: source_anim.channels[ch].target.node for ch, _, target in matches}
    target_rest, target_world, target_local = {}, {}, {}
    for i in target_order:
        parent = target_parents.get(i)
        rest = local_rest(model.nodes[i])
        target_rest[i] = target_rest.get(parent, Rotation.identity()) * rest
        parent_world = target_world.get(parent, Rotation.identity())
        if i in source_for_target:
            src = source_for_target[i]
            delta = src_world[src] * src_rest[src].inv()
            target_world[i] = delta * target_rest[i]
            target_local[i] = parent_world.inv() * target_world[i]
        else:
            target_world[i] = parent_world * rest

    bone_quats = {}
    for ch_idx, _, target in matches:
        data = target_local[target].as_quat()
        for frame_idx in range(1, len(data)):
            if np.dot(data[frame_idx - 1], data[frame_idx]) < 0:
                data[frame_idx] *= -1
        bone_quats[ch_idx] = data

    # Build timestamp and keyframe bytes
    timestamp_bytes = timestamps.tobytes()

    all_keyframes: list[np.ndarray] = []
    bone_order: list[str] = []
    for ch_idx, sampler_idx, node_idx in matches:
        bone_name = anim_gltf.nodes[source_anim.channels[ch_idx].target.node].name
        bone_order.append(bone_name)
        all_keyframes.append(bone_quats[ch_idx])

    all_keyframes_np = np.concatenate(all_keyframes, axis=0).astype(np.float32)
    keyframe_bytes = all_keyframes_np.tobytes()

    # Get the model's existing binary blob
    model_blob = model.binary_blob()
    if model_blob is None:
        model_blob = b""
    model_blob += b"\x00" * (-len(model_blob) % 4)
    model_blob_size = len(model_blob)

    # Append animation data after existing mesh data
    combined = model_blob + timestamp_bytes + keyframe_bytes

    # Update existing buffer to include new data
    if model.buffers:
        model.buffers[0].byteLength = len(combined)
    else:
        model.buffers.append(pygltflib.Buffer(byteLength=len(combined)))

    # Add buffer views for animation data (offsets relative to model blob)
    ts_bv_idx = len(model.bufferViews)
    model.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=model_blob_size,
            byteLength=len(timestamp_bytes),
        )
    )
    kf_bv_idx = len(model.bufferViews)
    model.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=model_blob_size + len(timestamp_bytes),
            byteLength=len(keyframe_bytes),
        )
    )

    # Timestamp accessor
    ts_acc_idx = len(model.accessors)
    model.accessors.append(
        pygltflib.Accessor(
            bufferView=ts_bv_idx,
            componentType=pygltflib.FLOAT,
            count=num_frames,
            type="SCALAR",
            max=[float(timestamps[-1])],
            min=[0.0],
        )
    )

    # Keyframe accessors
    kf_offset = 0
    kf_accessors: list[int] = []
    for bone_name in bone_order:
        acc_idx = len(model.accessors)
        model.accessors.append(
            pygltflib.Accessor(
                bufferView=kf_bv_idx,
                componentType=pygltflib.FLOAT,
                count=num_frames,
                type="VEC4",
                byteOffset=kf_offset,
            )
        )
        kf_accessors.append(acc_idx)
        kf_offset += num_frames * 4 * 4

    # Create animation on the model
    anim = pygltflib.Animation(name=source_anim.name or "mimic_animation")
    for i, bone_name in enumerate(bone_order):
        sampler_idx = len(anim.samplers)
        anim.samplers.append(
            pygltflib.AnimationSampler(
                input=ts_acc_idx,
                output=kf_accessors[i],
                interpolation="LINEAR",
            )
        )
        model_node_idx = model_nodes[bone_name]
        anim.channels.append(
            pygltflib.AnimationChannel(
                sampler=sampler_idx,
                target=pygltflib.AnimationChannelTarget(
                    node=model_node_idx,
                    path="rotation",
                ),
            )
        )
    model.animations.append(anim)

    # Set the combined binary blob
    model.set_binary_blob(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_binary(str(output_path))
    print(f"Saved: {output_path}")
    print(f"  {len(bone_order)} bones animated, {num_frames} frames @ {fps:.1f} fps")
    print(f"  File size: {output_path.stat().st_size / 1024:.0f} KB")
    return output_path


if __name__ == "__main__":
    import sys

    model = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/models/model.glb")
    anim = Path(sys.argv[2]) if len(sys.argv) > 2 else output_file("sneaky_walk.glb")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else output_file("sneaky_walk_animated.glb")
    merge_animation(model, anim, out)
