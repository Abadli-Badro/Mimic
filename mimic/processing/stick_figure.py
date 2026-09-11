"""Stick-figure animation for validating rotation solving."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mimic.processing.rotation_solver import BONE_DEFINITIONS


def _apply_rotation(point: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a point.

    Args:
        point: (3,) position.
        quat: (x, y, z, w) quaternion.

    Returns:
        Rotated (3,) position.
    """
    from scipy.spatial.transform import Rotation

    r = Rotation.from_quat(quat)
    return r.apply(point)


def render_stick_figure(
    landmarks: np.ndarray,
    frame_idx: int,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> None:
    """Render a stick figure from 3D landmarks.

    Args:
        landmarks: Array of shape (num_frames, 33, 3).
        frame_idx: Which frame to render.
        ax: Matplotlib 3D axes.
        title: Plot title.
    """
    lm = landmarks[frame_idx]

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Draw bones
    for parent_idx, child_idx, bone_name in BONE_DEFINITIONS:
        if parent_idx == child_idx:
            continue
        pts = np.array([lm[parent_idx], lm[child_idx]])
        ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], "b-", linewidth=2, alpha=0.7)

    # Draw joints
    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for idx in key_joints:
        ax.scatter(lm[idx, 0], lm[idx, 2], -lm[idx, 1], c="red", s=30)

    if title:
        ax.set_title(title)


def animate_stick_figure(
    landmarks: np.ndarray,
    output_path: Path,
    fps: float = 30.0,
    max_frames: int = 60,
    step: int = 1,
) -> Path:
    """Create a stick figure animation as a grid of frames.

    Args:
        landmarks: Array of shape (num_frames, 33, 3).
        output_path: Path to save the animation.
        fps: Frame rate.
        max_frames: Maximum frames to show.
        step: Frame step (2 = every other frame).

    Returns:
        Path to the saved image.
    """
    num_frames = landmarks.shape[0]
    indices = list(range(0, min(num_frames, max_frames * step), step))[:max_frames]

    cols = 6
    rows = (len(indices) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3, rows * 3))

    # Compute global bounds
    all_lm = landmarks.reshape(-1, 3)
    x_min, x_max = all_lm[:, 0].min(), all_lm[:, 0].max()
    z_min, z_max = all_lm[:, 2].min(), all_lm[:, 2].max()
    y_min, y_max = (-all_lm[:, 1]).min(), (-all_lm[:, 1]).max()
    margin = 0.1
    x_range = (x_min - margin, x_max + margin)
    y_range = (y_min - margin, y_max + margin)
    z_range = (z_min - margin, z_max + margin)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        render_stick_figure(landmarks, idx, ax=ax, title=f"Frame {idx}")
        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
