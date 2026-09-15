"""Structural checks for the subset of GLB supported by animation merging."""
import numpy as np
from mimic.errors import MimicError
from mimic.validation import hierarchy, rotations, array


def index(value, sequence, label):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(sequence):
        raise MimicError('invalid_gltf_reference', f'{label} references invalid index {value!r}.')
    return sequence[value]


def graph(gltf):
    if not gltf.nodes:
        raise MimicError('invalid_hierarchy', 'GLB contains no nodes.')
    parents = {i: None for i in range(len(gltf.nodes))}
    names = set()
    for i, node in enumerate(gltf.nodes):
        if node.name:
            name = node.name.replace('_', ':')
            if name in names:
                raise MimicError('ambiguous_bone_names', f'Duplicate node name after normalization: {name}')
            names.add(name)
        if node.matrix:
            raise MimicError('unsupported_transform', 'Decompose matrix node transforms to TRS before merging.')
        if node.rotation is not None:
            rotations({str(i): np.asarray(node.rotation)[None]})
        if node.translation is not None:
            array(node.translation, (3,), f'Node {i} translation')
        if node.scale is not None:
            scale = array(node.scale, (3,), f'Node {i} scale')
            if np.any(scale <= 0) or not np.allclose(scale, scale[0]):
                raise MimicError('unsupported_transform', 'Apply negative/nonuniform node scales before merging.')
        for child in node.children or []:
            index(child, gltf.nodes, f'Node {i} child')
            if parents[child] is not None:
                raise MimicError('invalid_hierarchy', f'Node {child} has multiple parent links.')
            parents[child] = i
    order = hierarchy(parents, single_root=False)
    for scene in gltf.scenes:
        for root in scene.nodes:
            index(root, gltf.nodes, 'Scene root')
            if parents[root] is not None:
                raise MimicError('invalid_hierarchy', 'A scene root also has a parent.')
    return parents, order


def buffers(gltf):
    blob = gltf.binary_blob() or b''
    if len(gltf.buffers) > 1 or any(b.uri for b in gltf.buffers):
        raise MimicError('unsupported_buffer', 'Merge requires a GLB with one embedded buffer.')
    if gltf.buffers and gltf.buffers[0].byteLength > len(blob):
        raise MimicError('truncated_buffer', 'GLB binary buffer is truncated.')
    for bv in gltf.bufferViews:
        if (bv.buffer != 0 or (bv.byteOffset or 0) < 0 or bv.byteLength < 0
                or (bv.byteOffset or 0) + bv.byteLength > len(blob)):
            raise MimicError('invalid_buffer_view', 'GLB buffer view lies outside its binary buffer.')
    return blob


def float_accessor(gltf, accessor_id, kind):
    a = index(accessor_id, gltf.accessors, 'Accessor')
    bv = index(a.bufferView, gltf.bufferViews, 'Accessor buffer view')
    if a.type != kind or a.componentType != 5126 or a.sparse is not None or a.normalized:
        raise MimicError('unsupported_accessor', f'Expected dense, unnormalized FLOAT {kind} accessor.')
    if not isinstance(a.count, int) or a.count < 1:
        raise MimicError('empty_animation', 'Accessor must contain at least one sample.')
    width = {'SCALAR': 1, 'VEC4': 4, 'MAT4': 16}[kind]
    offset = a.byteOffset or 0
    stride = bv.byteStride or width * 4
    if offset < 0 or offset % 4 or stride < width * 4 or stride % 4:
        raise MimicError('invalid_accessor', 'Accessor offset or stride is invalid.')
    if offset + (a.count - 1) * stride + width * 4 > bv.byteLength:
        raise MimicError('truncated_accessor', 'Accessor data exceeds its buffer view.')
    values = np.ndarray((a.count, width), dtype='<f4', buffer=gltf.binary_blob(),
                        offset=(bv.byteOffset or 0) + offset, strides=(stride,4)).copy()
    if not np.isfinite(values).all():
        raise MimicError('invalid_accessor', 'Accessor contains nonfinite values.')
    return values


def animation(gltf):
    if len(gltf.animations) != 1:
        raise MimicError('invalid_animation', 'Source GLB must contain exactly one animation.')
    anim = gltf.animations[0]
    if not anim.channels:
        raise MimicError('empty_animation', 'Animation contains no channels.')
    nodes = set()
    for ch in anim.channels:
        index(ch.target.node, gltf.nodes, 'Animation node')
        sampler = index(ch.sampler, anim.samplers, 'Animation sampler')
        if ch.target.path != 'rotation' or sampler.interpolation not in (None, 'LINEAR'):
            raise MimicError('unsupported_animation', 'Merge accepts LINEAR rotation-only animation.')
        if ch.target.node in nodes:
            raise MimicError('duplicate_channel', 'Animation has duplicate rotation channels for a node.')
        nodes.add(ch.target.node)
        times = float_accessor(gltf, sampler.input, 'SCALAR')[:,0]
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise MimicError('invalid_timeline', 'Animation times must be nonnegative and strictly increasing.')
        q = float_accessor(gltf, sampler.output, 'VEC4')
        rotations({str(ch.target.node): q}, num_frames=len(times))


def skins(gltf):
    if not gltf.skins:
        raise MimicError('missing_skin', 'Target model has no skin. Supply a rigged GLB character.')
    for skin in gltf.skins:
        if not skin.joints or len(set(skin.joints)) != len(skin.joints):
            raise MimicError('invalid_skin', 'Skin joints must be nonempty and unique.')
        for joint in skin.joints:
            index(joint, gltf.nodes, 'Skin joint')
        if skin.skeleton is not None:
            index(skin.skeleton, gltf.nodes, 'Skin skeleton')
        if skin.inverseBindMatrices is not None:
            matrices = float_accessor(gltf, skin.inverseBindMatrices, 'MAT4')
            if len(matrices) != len(skin.joints):
                raise MimicError('invalid_skin', 'Inverse bind matrix count does not match skin joints.')
    for node in gltf.nodes:
        if node.skin is not None:
            index(node.skin, gltf.skins, 'Node skin')
        if node.mesh is not None:
            index(node.mesh, gltf.meshes, 'Node mesh')


def load_glb(path):
    """Check the container before asking pygltflib to construct its objects."""
    import json
    import struct
    import pygltflib
    from pathlib import Path
    path = Path(path)
    try:
        with path.open('rb') as stream:
            header = stream.read(20)
            if len(header) != 20:
                raise ValueError('Incomplete GLB header')
            magic, version, length, json_length, kind = struct.unpack('<4sIIII', header)
            if (magic != b'glTF' or version != 2 or length != path.stat().st_size
                    or kind != 0x4E4F534A or json_length > length - 20 or json_length % 4):
                raise ValueError('Invalid GLB header or chunk length')
            doc = json.loads(stream.read(json_length))
            if not isinstance(doc, dict) or not isinstance(doc.get('asset'), dict):
                raise ValueError('GLB JSON must contain an asset object')
            if doc['asset'].get('version') != '2.0':
                raise ValueError('Expected glTF asset version 2.0')
        return pygltflib.GLTF2.load(str(path))
    except (ValueError, TypeError, KeyError, AttributeError, struct.error) as exc:
        raise MimicError('invalid_glb', f'Cannot read GLB {path}: {exc}') from exc
