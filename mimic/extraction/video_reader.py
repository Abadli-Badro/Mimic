"""Bounded OpenCV video decoding with resource and metadata validation."""
from pathlib import Path
import cv2
import numpy as np
from mimic.config import VIDEO_LIMITS
from mimic.errors import MimicError
from mimic.validation import number


def get_video_info(video_path: Path, limits=None) -> dict:
    limits = limits or VIDEO_LIMITS
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f'Video file not found: {video_path}')
    size = video_path.stat().st_size
    if not size:
        raise MimicError('empty_video', f'Video is empty: {video_path}')
    if size > limits.max_file_bytes:
        raise MimicError('video_too_large', f'Video exceeds {limits.max_file_bytes / 1024**2:g} MiB. Compress or trim it.')
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise MimicError('video_decode', 'Cannot open video. Check the file and re-encode it as H.264 MP4.')
        fps = number(cap.get(cv2.CAP_PROP_FPS), 'video fps', maximum=limits.max_fps)
        width = number(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'video width')
        height = number(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'video height')
        count = number(cap.get(cv2.CAP_PROP_FRAME_COUNT), 'video frame count', maximum=limits.max_frames)
        if not all(v.is_integer() for v in (width, height, count)):
            raise MimicError('video_metadata', 'Invalid video dimensions or frame count. Re-encode the video.')
        if width * height > limits.max_pixels or max(width, height) > limits.max_dimension:
            raise MimicError('video_resolution', f'Video exceeds {limits.max_pixels:,} pixels per frame. Downscale it.')
        duration = count / fps
        if duration > limits.max_duration_seconds:
            raise MimicError('video_duration', f'Video exceeds {limits.max_duration_seconds:g} seconds. Trim it.')
        return dict(fps=fps, width=int(width), height=int(height), total_frames=int(count), duration=duration)
    finally:
        cap.release()


def read_frames(video_path: Path, target_fps: int | None = None, limits=None):
    limits = limits or VIDEO_LIMITS
    info = get_video_info(video_path, limits)
    fps = info['fps']
    if target_fps is not None:
        number(target_fps, 'target fps', maximum=limits.max_fps)
    step = max(1, round(fps / target_fps)) if target_fps and target_fps < fps else 1
    estimated = ((info['total_frames'] + step - 1) // step) * info['width'] * info['height'] * 3
    if estimated > limits.max_decoded_bytes:
        raise MimicError('video_memory_limit', 'Decoded video exceeds the memory budget. Trim or downscale the clip.')
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = used = 0
    try:
        if not cap.isOpened():
            raise MimicError('video_decode', f'Cannot open video: {video_path}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape != (info['height'], info['width'], 3) or frame.dtype != np.uint8:
                raise MimicError('invalid_video_frame', f'Invalid decoded frame {count}. Re-encode the video.')
            count += 1
            if count > limits.max_frames or count / fps > limits.max_duration_seconds:
                raise MimicError('video_duration', 'Decoded video exceeds the frame or duration limit.')
            if (count - 1) % step == 0:
                used += frame.nbytes
                if used > limits.max_decoded_bytes:
                    raise MimicError('video_memory_limit', 'Decoded video exceeds the memory budget. Trim or downscale it.')
                frames.append(frame)
        if not count:
            raise MimicError('video_decode', 'Video contains no decodable frames.')
        if count != info['total_frames']:
            raise MimicError('truncated_video', f'Decoded {count} of {info["total_frames"]} advertised frames. Re-encode the video.')
    finally:
        cap.release()
    return frames, fps / step
