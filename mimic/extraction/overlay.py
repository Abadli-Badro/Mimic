"""Generate overlay video with skeleton drawn on original frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 30),
    (24, 26), (26, 28), (28, 30), (28, 31),
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (0, 8),
    (9, 10),
]

COLORS = {
    "line": (0, 255, 0),
    "dot": (0, 0, 255),
    "text": (255, 255, 255),
}


def draw_skeleton(
    frame: np.ndarray,
    landmarks_2d: np.ndarray,
    visibility: np.ndarray,
    width: int,
    height: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Draw skeleton overlay on a frame.

    Args:
        frame: BGR image.
        landmarks_2d: Normalized (x, y) landmarks, shape (33, 2).
        visibility: Per-landmark visibility, shape (33,).
        width: Frame width in pixels.
        height: Frame height in pixels.
        threshold: Minimum visibility to draw.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    def to_px(idx: int) -> tuple[int, int] | None:
        if visibility[idx] < threshold:
            return None
        x = int(landmarks_2d[idx, 0] * width)
        y = int(landmarks_2d[idx, 1] * height)
        return (x, y)

    for i, j in POSE_CONNECTIONS:
        p1 = to_px(i)
        p2 = to_px(j)
        if p1 and p2:
            cv2.line(annotated, p1, p2, COLORS["line"], 2, cv2.LINE_AA)

    for idx in range(33):
        pt = to_px(idx)
        if pt:
            cv2.circle(annotated, pt, 4, COLORS["dot"], -1, cv2.LINE_AA)

    return annotated


def generate_overlay_video(
    video_path: Path,
    npz_path: Path,
    output_path: Path | None = None,
    fps: float | None = None,
    threshold: float = 0.5,
) -> Path:
    """Generate a video with skeleton overlay.

    Args:
        video_path: Path to the original MP4 video.
        npz_path: Path to the extracted landmarks .npz file.
        output_path: Path for the output video. Defaults to video_name_overlay.mp4.
        fps: Override fps. If None, uses source video fps.
        threshold: Minimum visibility to draw a landmark.

    Returns:
        Path to the generated overlay video.
    """
    if output_path is None:
        output_path = video_path.with_name(video_path.stem + "_overlay.mp4")

    data = dict(np.load(npz_path, allow_pickle=True))
    landmarks_2d = data["landmarks_2d"]
    visibility = data["visibility"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = fps or cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < landmarks_2d.shape[0]:
            annotated = draw_skeleton(
                frame, landmarks_2d[frame_idx], visibility[frame_idx],
                width, height, threshold,
            )
        else:
            annotated = frame

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path


if __name__ == "__main__":
    import sys

    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input/Sneaky walk reference.mp4")
    npz = video.with_suffix(".npz")
    out = generate_overlay_video(video, npz)
    print(f"Overlay video saved to: {out}")
