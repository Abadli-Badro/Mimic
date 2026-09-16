"""Deduplicate pose proposals and confirm distinct people over time."""
from __future__ import annotations

import numpy as np
from mimic.errors import MimicError


class PoseSelector:
    """Track the main person; overlapping proposals are not separate people."""

    def __init__(self, fps, confirmation_seconds=0.3):
        self.required_frames = max(3, int(np.ceil(fps * confirmation_seconds)))
        self.multiple_frames = 0
        self.previous = None

    def select(self, poses, frame_index, timestamp_ms, aspect_ratio=1.0):
        candidates = []
        for index, pose in enumerate(poses):
            if len(pose) != 33:
                continue
            core = [pose[i] for i in (11, 12, 23, 24)]
            points = np.array([[p.x * aspect_ratio, p.y] for p in core])
            confidence = np.array([[p.visibility, getattr(p, 'presence', 1.0)] for p in core])
            if not np.isfinite(points).all() or not np.isfinite(confidence).all():
                continue
            reliable = (confidence[:, 0] >= .6) & (confidence[:, 1] >= .8)
            scale = np.linalg.norm(points[:2].mean(axis=0) - points[2:].mean(axis=0))
            if reliable.sum() < 3 or scale < .02:
                continue
            score = float(confidence.mean())
            candidates.append((index, points, scale, score))

        # Confidence-ranked nonmaximum suppression using corresponding torso
        # landmarks. Bounding boxes alone can overlap for two real people.
        distinct = []
        for candidate in sorted(candidates, key=lambda c: c[3], reverse=True):
            _, points, scale, _ = candidate
            duplicate = any(np.linalg.norm(points - other[1], axis=1).mean()
                            < .35 * max(scale, other[2]) for other in distinct)
            if not duplicate:
                distinct.append(candidate)

        self.multiple_frames = self.multiple_frames + 1 if len(distinct) > 1 else 0
        if self.multiple_frames >= self.required_frames:
            raise MimicError('multiple_people',
                             f'Multiple distinct, reliable poses persisted for {self.multiple_frames} '
                             f'frames at frame {frame_index} ({timestamp_ms / 1000:.2f}s). '
                             'Use a clip showing only one person.')
        if not distinct:
            return None
        if self.previous is None:
            chosen = distinct[0]
        else:
            chosen = min(distinct, key=lambda c: np.linalg.norm(c[1] - self.previous, axis=1).mean())
        self.previous = chosen[1]
        return chosen[0]
