"""Merge skeleton animation onto a full Mixamo rigged model (mesh + skin)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation


def merge_animation(
    model_path: Path,
    animation_path: Path,
    output_path: Path,
) -> Path:
    """Take animation keyframes from one GLB and apply them to another GLB's matching nodes.

    Pre-multiplies each bone's animation rotation by the model's rest rotation,
    so that the animation correctly replaces the node's rotation in glTF
    (where animation channels OVERRIDE node.rotation, they don't combine with it).

    Also strips any existing animations from the model to avoid conflicts.

    Args:
        model_path: Full rigged model .glb (mesh, skin, T-pose skeleton).
        animation_path: Skeleton-only .glb with rotation animation.
        output_path: Output .glb with full mesh + animation.

    Returns:
        Path to output file.
    """
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
        if src_node.name in model_nodes:
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
    ts_offset = ts_bv.byteOffset + (ts_acc.byteOffset or 0)
    timestamps = np.frombuffer(
        src_blob[ts_offset:ts_offset + ts_acc.count * 4],
        dtype=np.float32,
    )
    num_frames = ts_acc.count
    fps = float(num_frames - 1) / float(timestamps[-1]) if timestamps[-1] > 0 else 30.0

    # Read per-bone keyframe data, pre-multiply by model rest rotations
    bone_quats: dict[int, np.ndarray] = {}
    rest_count = 0
    for ch_idx, sampler_idx, model_node_idx in matches:
        sampler = source_anim.samplers[sampler_idx]
        out_acc = anim_gltf.accessors[sampler.output]
        out_bv = anim_gltf.bufferViews[out_acc.bufferView]
        out_offset = out_bv.byteOffset + (out_acc.byteOffset or 0)
        data = np.frombuffer(
            src_blob[out_offset:out_offset + out_acc.count * 16],
            dtype=np.float32,
        ).reshape(out_acc.count, 4).copy()

        # Pre-multiply by model rest rotation
        model_node = model.nodes[model_node_idx]
        rest_q = np.array(model_node.rotation if model_node.rotation else [0.0, 0.0, 0.0, 1.0])
        if not np.allclose(rest_q, [0, 0, 0, 1], atol=0.001):
            rest_rot = Rotation.from_quat(rest_q)
            for frame_idx in range(data.shape[0]):
                solver_q = data[frame_idx]
                final_rot = rest_rot * Rotation.from_quat(solver_q)
                data[frame_idx] = final_rot.as_quat()
            rest_count += 1

        bone_quats[ch_idx] = data.copy()

    print(f"Pre-multiplied rest rotations for {rest_count} bones")

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
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    kf_bv_idx = len(model.bufferViews)
    model.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=model_blob_size + len(timestamp_bytes),
            byteLength=len(keyframe_bytes),
            target=pygltflib.ARRAY_BUFFER,
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
    anim = pygltflib.Animation(name="sneaky_walk")
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
    anim = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/input/sneaky_walk.glb")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/output/sneaky_walk_animated.glb")
    merge_animation(model, anim, out)
