"""Human-pose usability checks; this is not an animal/species classifier."""
import numpy as np
from mimic.config import POSE_QUALITY
from mimic.errors import MimicError


def validate_pose_quality(world, visibility, fps, policy=None, enforce_continuity=True):
    policy = policy or POSE_QUALITY
    visible = visibility >= policy.visibility_threshold
    core = [11,12,23,24,25,26,27,28]
    reliable = (visible[:,core].sum(axis=1) >= 4) & (visible[:,[11,12,23,24]].sum(axis=1) >= 3)
    if not reliable.any():
        raise MimicError('no_human_pose', 'No person detected reliably. Use a clear, well-lit video with one visible human body; animals are unsupported.')
    first, last = np.flatnonzero(visibility.sum(axis=1) > 1)[[0,-1]]
    retained = reliable[first:last+1]
    torso = np.linalg.norm((world[:,11]+world[:,12]-world[:,23]-world[:,24])*.5,axis=1)
    shoulders = np.linalg.norm(world[:,11]-world[:,12],axis=1)
    hips = np.linalg.norm(world[:,23]-world[:,24],axis=1)
    plausible = (torso > .08) & (torso < 1.5) & (shoulders > .03) & (hips > .02)
    good = reliable & plausible
    fraction = good[first:last+1].mean() if enforce_continuity else good.sum() / reliable.sum()
    if fraction < policy.minimum_good_fraction:
        raise MimicError('poor_pose_quality', 'Too few reliable body poses or collapsed body geometry. Improve lighting, framing, or use a clearer clip.')
    if good.sum() < policy.minimum_good_frames:
        raise MimicError('insufficient_motion', f'Need at least {policy.minimum_good_frames} reliable pose frames. Use a longer visible segment.')
    longest = gap = 0
    for ok in good[first:last+1]:
        gap = 0 if ok else gap + 1
        longest = max(longest, gap)
    if enforce_continuity and (retained.mean() < policy.minimum_good_fraction or longest / fps > policy.maximum_gap_seconds):
        raise MimicError('tracking_lost', 'Pose tracking is unreliable or missing for too long inside the clip. Trim the gap or use clearer footage.')
    return int(first), int(last)
