"""Shared data contracts for pipeline boundaries."""
import numpy as np
from mimic.errors import MimicError


def number(value, name, minimum=0, maximum=None, inclusive=False):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in 'iuf' or not np.isfinite(a):
        raise MimicError('invalid_parameter', f'{name} must be a finite number.')
    result = float(a)
    if (result < minimum if inclusive else result <= minimum) or (maximum is not None and result > maximum):
        comparison = '>=' if inclusive else '>'
        raise MimicError('invalid_parameter', f'{name} must be {comparison} {minimum}' +
                         (f' and <= {maximum}.' if maximum is not None else '.'))
    return result


def array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise MimicError('invalid_array', f'{name} must contain finite numbers with shape {shape}.')
    return a


def landmarks(data, overlay=False):
    required = {'landmarks_2d', 'visibility'} if overlay else {
        'world_landmarks', 'landmarks_2d', 'visibility', 'fps', 'landmark_names'}
    missing = required - data.keys()
    if missing:
        raise MimicError('missing_fields', f'Landmark file is missing: {", ".join(sorted(missing))}. Run extract again.')
    key = 'landmarks_2d' if overlay else 'world_landmarks'
    a = np.asarray(data[key])
    if a.ndim != 3 or not len(a):
        raise MimicError('invalid_landmarks', 'Landmark file must contain at least one frame.')
    n = len(a)
    if not overlay:
        array(a, (n,33,3), key)
        number(data['fps'], 'fps', maximum=240)
        if np.shape(data['landmark_names']) != (33,):
            raise MimicError('invalid_landmarks', 'Expected 33 landmark names.')
    array(data['landmarks_2d'], (n,33,2), 'landmarks_2d')
    vis = array(data['visibility'], (n,33), 'visibility')
    if np.any((vis < 0) | (vis > 1)):
        raise MimicError('invalid_visibility', 'Visibility values must be between 0 and 1.')
    first = number(data.get('first_frame', 0), 'first_frame', inclusive=True)
    if not first.is_integer():
        raise MimicError('invalid_landmarks', 'first_frame must be an integer.')
    return n


def hierarchy(parents, single_root=True):
    if not isinstance(parents, dict) or not parents:
        raise MimicError('invalid_hierarchy', 'Skeleton hierarchy must be a nonempty mapping.')
    children = {name: [] for name in parents}
    roots = []
    for name, parent in parents.items():
        if parent is None:
            roots.append(name)
        elif parent not in parents or parent == name:
            raise MimicError('invalid_hierarchy', f'Joint {name!r} has an invalid parent: {parent!r}.')
        else:
            children[parent].append(name)
    if not roots or (single_root and len(roots) != 1):
        raise MimicError('invalid_hierarchy', 'Skeleton must have exactly one root.' if single_root else 'Node graph has no root.')
    order = []
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    if len(order) != len(parents):
        raise MimicError('invalid_hierarchy', 'Skeleton contains a cycle or a disconnected cyclic component.')
    return order


def rotations(tracks, parents=None, num_frames=None):
    if not isinstance(tracks, dict) or not tracks:
        raise MimicError('empty_animation', 'Animation has no rotation tracks.')
    if parents is not None:
        hierarchy(parents)
        unknown = tracks.keys() - parents.keys()
        if unknown:
            raise MimicError('unknown_bones', f'Animation contains unknown bones: {sorted(unknown)}.')
    count = None
    for name, values in tracks.items():
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != 4 or not len(a):
            raise MimicError('invalid_rotations', f'{name}: expected a nonempty (frames, 4) quaternion track.')
        if count is None:
            count = len(a)
        array(a, (count,4), name)
        norms = np.linalg.norm(a,axis=1)
        if np.any(np.abs(norms - 1) > .01):
            raise MimicError('invalid_quaternion', f'{name}: quaternions must have unit length (x, y, z, w).')
    if num_frames is not None and number(num_frames, 'num_frames') != count:
        raise MimicError('frame_count_mismatch', f'Expected {num_frames} frames but tracks contain {count}.')
    return count
