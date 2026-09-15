"""Before/after trajectory plots for validating smoothing quality."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mimic.extraction.pose_extractor import LANDMARK_NAMES


def plot_trajectory_comparison(
    raw: np.ndarray,
    smoothed: np.ndarray,
    landmark_idx: int,
    dim: int = 0,
    fps: float = 30.0,
    title: str | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot raw vs smoothed trajectory for a single landmark dimension.

    Args:
        raw: Raw landmarks, shape (num_frames, 33, 3).
        smoothed: Smoothed landmarks, shape (num_frames, 33, 3).
        landmark_idx: Which landmark to plot (0-32).
        dim: Which dimension (0=x, 1=y, 2=z).
        fps: Frame rate.
        title: Plot title.
        save_path: If provided, save to this path.
    """
    num_frames = raw.shape[0]
    t = np.arange(num_frames) / fps
    dim_names = ["X", "Y", "Z"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw[:, landmark_idx, dim], "r-", alpha=0.5, linewidth=0.8, label="Raw")
    ax.plot(
        t, smoothed[:, landmark_idx, dim], "b-", linewidth=1.5, label="Smoothed"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{dim_names[dim]} (m)")
    ax.set_title(
        title or f"{LANDMARK_NAMES[landmark_idx]} — {dim_names[dim]} trajectory"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_all_trajectories(
    raw: np.ndarray,
    smoothed: np.ndarray,
    output_dir: Path,
    fps: float = 30.0,
    landmark_indices: list[int] | None = None,
    dims: list[int] | None = None,
) -> list[Path]:
    """Plot before/after for multiple landmark dimensions.

    Args:
        raw: Raw landmarks (num_frames, 33, 3).
        smoothed: Smoothed landmarks (num_frames, 33, 3).
        output_dir: Directory to save plots.
        fps: Frame rate.
        landmark_indices: Which landmarks to plot. Default: major joints.
        dims: Which dimensions to plot. Default: all (0, 1, 2).

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if landmark_indices is None:
        landmark_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if dims is None:
        dims = [0, 1, 2]

    saved = []
    for lm_idx in landmark_indices:
        for dim in dims:
            path = output_dir / f"{LANDMARK_NAMES[lm_idx]}_{['x','y','z'][dim]}.png"
            plot_trajectory_comparison(
                raw, smoothed, lm_idx, dim, fps, save_path=path,
            )
            saved.append(path)

    return saved
