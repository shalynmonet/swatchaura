"""Small OpenCV helpers for reading and writing video clips.

Frames use OpenCV's BGR color order. Keeping that convention here avoids extra
color conversions in later image-processing steps.
"""

from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def read_video(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read every frame from a video and return ``(frames, fps)``.

    Args:
        video_path: Path to a video file on disk. Streamlit uploads can be
            saved to a temporary file before calling this function.

    Raises:
        FileNotFoundError: If the path does not point to a file.
        ValueError: If OpenCV cannot open the file, or it has no usable frames
            or frame rate.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file was not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open this video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"The video contains no readable frames: {path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"The video has an invalid frame rate: {fps}")

    return frames, fps


def write_video(
    frames: list[np.ndarray], output_path: str | Path, fps: float
) -> Path:
    """Write BGR frames to an MP4 file at the supplied frame rate.

    All frames must have the same height and width. The returned path is handy
    for opening the finished video or exposing it as a download.
    """
    if not frames:
        raise ValueError("At least one frame is required to write a video.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Frame rate must be a positive number, got {fps}.")

    first = frames[0]
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Frames must be color images with three BGR channels.")
    height, width = first.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("Frames must have non-zero width and height.")

    for index, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape != first.shape:
            raise ValueError(
                f"Frame {index} has shape {frame.shape}; expected {first.shape}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Frame {index} has dtype {frame.dtype}; expected uint8."
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # MPEG-4 Part 2 (`mp4v`) is not supported by many browser video players.
    # H.264 baseline in yuv420p is broadly playable in browsers and Windows.
    encoder = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=float(fps),
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=1,
        output_params=[
            "-preset",
            "veryfast",
            "-profile:v",
            "baseline",
            "-movflags",
            "+faststart",
        ],
    )
    try:
        encoder.send(None)
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encoder.send(np.ascontiguousarray(rgb_frame).tobytes())
    finally:
        encoder.close()

    return path
