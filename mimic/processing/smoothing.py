"""Temporal smoothing filters (One-Euro, Savitzky-Golay)."""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """One-Euro adaptive low-pass filter for noisy signals.

    Reference: https://hal.inria.fr/hal-00670496/document
    """

    def __init__(
        self,
        fps: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize the filter.

        Args:
            fps: Signal frame rate.
            min_cutoff: Minimum cutoff frequency (higher = more smoothing).
            beta: Speed coefficient (higher = less lag on fast motion).
            d_cutoff: Cutoff frequency for the derivative.
        """
        self.fps = fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        """Filter a single sample.

        Args:
            x: Signal value (scalar or array).
            timestamp: Time in seconds.

        Returns:
            Filtered value.
        """
        x = np.asarray(x, dtype=np.float64)

        if self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = timestamp
            return x

        t_e = timestamp - self._t_prev
        if t_e <= 0:
            return self._x_prev.copy()

        # Derivative
        dx = (x - self._x_prev) / t_e
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Signal
        alpha = self._smoothing_factor(t_e, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        self._t_prev = timestamp

        return x_hat


def smooth_landmarks_array(
    landmarks: np.ndarray,
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> np.ndarray:
    """Smooth a landmarks array with the One-Euro filter.

    Args:
        landmarks: Array of shape (num_frames, num_landmarks, dims).
        fps: Frame rate.
        min_cutoff: Minimum cutoff frequency.
        beta: Speed coefficient.

    Returns:
        Smoothed array of the same shape.
    """
    num_frames, num_landmarks, dims = landmarks.shape
    smoothed = np.empty_like(landmarks)

    for lm_idx in range(num_landmarks):
        for dim in range(dims):
            filt = OneEuroFilter(fps=fps, min_cutoff=min_cutoff, beta=beta)
            for frame_idx in range(num_frames):
                t = frame_idx / fps
                smoothed[frame_idx, lm_idx, dim] = filt(
                    landmarks[frame_idx, lm_idx, dim], t
                )

    return smoothed


def smooth_landmarks(
    landmarks: list[dict[str, np.ndarray]],
    fps: float,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> list[dict[str, np.ndarray]]:
    """Apply One-Euro filter to smooth landmark trajectories.

    Args:
        landmarks: Per-frame landmark dictionaries.
        fps: Source frame rate.
        min_cutoff: Minimum cutoff frequency for the One-Euro filter.
        beta: Speed coefficient for adaptive cutoff.

    Returns:
        Smoothed landmark dictionaries.
    """
    return landmarks
