"""Generate overlay video with skeleton drawn on original frames."""

from __future__ import annotations

from pathlib import Path

from mimic.config import output_file
from mimic.errors import stage, MimicError
from mimic.artifacts import load_npz, atomic_output, output_path as check_output
from mimic.validation import landmarks, number
from mimic.extraction.video_reader import get_video_info

import cv2
import numpy as np

from mimic.extraction.pose_extractor import POSE_CONNECTIONS

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


@stage("overlay export")
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
        output_path = output_file(video_path.stem + "_overlay.mp4")

    output_path = check_output(output_path, '.mp4', [video_path, npz_path])
    number(threshold, 'visibility threshold', maximum=1, inclusive=True)
    data = load_npz(npz_path)
    landmarks(data, overlay=True)
    landmarks_2d, visibility = data['landmarks_2d'], data['visibility']
    if np.any(np.abs(landmarks_2d) > 10):
        raise MimicError('invalid_landmarks', 'Overlay coordinates must be normalized image coordinates.')
    info = get_video_info(video_path)
    first_frame = int(data.get('first_frame', 0))
    if first_frame + len(landmarks_2d) > info['total_frames']:
        raise MimicError('frame_count_mismatch', 'Landmark sequence extends beyond the source video.')
    width, height = info['width'], info['height']
    source_fps = number(fps if fps is not None else info['fps'], 'fps', maximum=240)
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            raise MimicError('video_decode', f'Cannot open video: {video_path}')
        with atomic_output(output_path) as temporary:
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'),
                                     source_fps, (width, height))
            try:
                if not writer.isOpened():
                    raise MimicError('video_encode', 'Cannot initialize MP4 encoder. Check codec availability and output permissions.')
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame is None or frame.shape != (height,width,3):
                        raise MimicError('video_decode', f'Invalid video frame {frame_idx}.')
                    pose_idx = frame_idx - first_frame
                    if 0 <= pose_idx < len(landmarks_2d):
                        frame = draw_skeleton(frame, landmarks_2d[pose_idx], visibility[pose_idx],
                                              width, height, threshold)
                    writer.write(frame)
                    frame_idx += 1
                    if frame_idx > info['total_frames']:
                        raise MimicError('video_metadata', 'Video frame count changed during overlay export.')
            finally:
                writer.release()
                writer = None
            if frame_idx != info['total_frames']:
                raise MimicError('truncated_video', 'Video decoding stopped early during overlay export.')
            check = cv2.VideoCapture(str(temporary))
            try:
                if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_idx:
                    raise MimicError('video_encode', 'MP4 encoder did not write all overlay frames.')
            finally:
                check.release()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return output_path


if __name__ == "__main__":
    import sys

    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/input/Sneaky walk reference.mp4")
    npz = output_file(video.stem + ".npz")
    out = generate_overlay_video(video, npz)
    print(f"Overlay video saved to: {out}")
