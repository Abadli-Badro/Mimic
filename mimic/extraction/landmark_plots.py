"""3D landmark visualization for sanity-checking pose tracking."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mimic.extraction.pose_extractor import LANDMARK_NAMES, POSE_CONNECTIONS


def plot_landmarks_3d(
    landmarks: np.ndarray,
    frame_idx: int = 0,
    title: str | None = None,
    save_path: Path | None = None,
    visibility: np.ndarray | None = None,
    visibility_threshold: float = 0.5,
) -> None:
    """Plot 3D pose landmarks for a single frame.

    Args:
        landmarks: Array of shape (num_frames, 33, 3).
        frame_idx: Which frame to plot.
        title: Plot title.
        save_path: If provided, save the figure to this path.
        visibility: Optional visibility array of shape (num_frames, 33).
        visibility_threshold: Minimum visibility to show a landmark.
    """
    lm = landmarks[frame_idx]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    vis = visibility[frame_idx] if visibility is not None else np.ones(33)

    # Plot connections
    for i, j in POSE_CONNECTIONS:
        if vis[i] > visibility_threshold and vis[j] > visibility_threshold:
            ax.plot(
                [lm[i, 0], lm[j, 0]],
                [lm[i, 1], lm[j, 1]],
                [lm[i, 2], lm[j, 2]],
                "b-",
                alpha=0.6,
            )

    # Plot landmarks
    for idx in range(33):
        if vis[idx] > visibility_threshold:
            ax.scatter(lm[idx, 0], lm[idx, 1], lm[idx, 2], c="red", s=20)
            ax.text(lm[idx, 0], lm[idx, 1], lm[idx, 2], LANDMARK_NAMES[idx], fontsize=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or f"Frame {frame_idx}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_landmarks_all_frames(
    landmarks: np.ndarray,
    output_dir: Path,
    visibility: np.ndarray | None = None,
    max_frames: int = 10,
    visibility_threshold: float = 0.5,
) -> list[Path]:
    """Plot and save 3D scatter for multiple frames.

    Args:
        landmarks: Array of shape (num_frames, 33, 3).
        output_dir: Directory to save images.
        visibility: Optional visibility array.
        max_frames: Maximum number of frames to plot.
        visibility_threshold: Minimum visibility to show a landmark.

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = landmarks.shape[0]
    step = max(1, num_frames // max_frames)
    saved = []

    for i in range(0, num_frames, step):
        if len(saved) >= max_frames:
            break
        path = output_dir / f"frame_{i:04d}.png"
        plot_landmarks_3d(
            landmarks,
            frame_idx=i,
            save_path=path,
            visibility=visibility,
            visibility_threshold=visibility_threshold,
        )
        saved.append(path)

    return saved
