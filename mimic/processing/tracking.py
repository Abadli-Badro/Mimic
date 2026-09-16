"""Bounded per-bone tracking gaps and a shared animation cutoff."""
from __future__ import annotations
import numpy as np
from mimic.config import MAX_MISSING_BONE_FRAMES, BONE_MISSING_FRAME_LIMITS
from mimic.errors import MimicError


def tracking_window(landmarks, visibility=None, threshold=.5,
                    max_missing_frames=MAX_MISSING_BONE_FRAMES, bone_limits=None):
    from mimic.processing.rotation_solver import (
        BONE_ORDER, BONE_LANDMARKS, _compute_bone_direction, _body_frame,
    )
    overrides = dict(BONE_MISSING_FRAME_LIMITS)
    if bone_limits is not None:
        overrides.update(bone_limits)
    if set(overrides) - set(BONE_ORDER):
        raise MimicError('unknown_bones', f'Unknown timer bones: {sorted(set(overrides)-set(BONE_ORDER))}')
    limits = {b: overrides.get(b, max_missing_frames) for b in BONE_ORDER}
    for name, value in {'default':max_missing_frames, **limits}.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise MimicError('invalid_parameter', f'Missing-frame limit for {name} must be a nonnegative integer.')
    observations = {}
    for bone in BONE_ORDER:
        indices = BONE_LANDMARKS[bone]
        valid = np.isfinite(landmarks[:,indices]).all(axis=(1,2))
        if visibility is not None:
            valid &= (visibility[:,indices] >= threshold).all(axis=1)
        for frame in np.flatnonzero(valid):
            lm = landmarks[frame]
            direction = _compute_bone_direction(bone, lm)
            if bone in {'pelvis','spine','spine1','chest'}:
                lateral = lm[23]-lm[24] if bone == 'pelvis' else lm[11]-lm[12]
                valid[frame] = _body_frame(lateral,direction) is not None
            elif bone == 'head':
                lateral = lm[7]-lm[8]
                forward = lm[0]-(lm[7]+lm[8])*.5
                valid[frame] = _body_frame(lateral,np.cross(forward,lateral)) is not None
            else:
                valid[frame] = np.linalg.norm(direction) > 1e-8
        observations[bone] = valid
    missing = {bone:0 for bone in BONE_ORDER}
    stop_frame = len(landmarks)
    stopped_bones = []
    for frame in range(len(landmarks)):
        for bone in BONE_ORDER:
            missing[bone] = 0 if observations[bone][frame] else missing[bone]+1
        stopped_bones = [bone for bone in BONE_ORDER if missing[bone] > limits[bone]]
        if stopped_bones:
            stop_frame = frame
            break
    if stop_frame == 0:
        raise MimicError('tracking_timeout', f'No exportable frames: timers expired for {stopped_bones} at frame 0.')
    return {'num_frames':stop_frame, 'input_frames':len(landmarks),
            'stopped_bones':stopped_bones, 'observations':observations, 'limits':limits}
