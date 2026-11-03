"""Per-frame object masks with a local GrabCut/optical-flow mock.

The mock starts from a rough rectangle on the first frame, segments that
region with GrabCut, then follows the mask between frames using sparse optical
flow and a robust affine estimate. Results are cached by video, prompt, mode,
and starting rectangle so reruns are quick and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import mock_mode_enabled
from .meta_client import segment_video_with_sam31

Box = tuple[int, int, int, int]
DEFAULT_MASK_CACHE = Path("cache") / "masks"


def _validate_frames(frames: list[np.ndarray]) -> tuple[int, int]:
    if not frames:
        raise ValueError("At least one video frame is required for segmentation.")
    first = frames[0]
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Frames must be uint8 BGR color images.")
    height, width = first.shape[:2]
    for index, frame in enumerate(frames):
        if frame.dtype != np.uint8 or frame.shape != first.shape:
            raise ValueError(
                f"Frame {index} has shape/dtype {frame.shape}/{frame.dtype}; "
                f"expected {first.shape}/uint8."
            )
    return width, height


def _video_fingerprint(frames: list[np.ndarray]) -> str:
    """Hash all frame bytes so changed clips cannot reuse stale masks."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(str(frame.shape).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(frame)))
    return digest.hexdigest()


def _cache_path(
    frames: list[np.ndarray],
    object_prompt: str,
    mock_mode: bool,
    first_frame_box: Box | None,
    cache_dir: str | Path,
) -> Path:
    identity = {
        "video": _video_fingerprint(frames),
        "prompt": object_prompt.strip(),
        "mock_mode": mock_mode,
        "box": first_frame_box,
        "version": 1,
    }
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Path(cache_dir) / f"{key}.npz"


def _read_cached_masks(path: Path, frame_count: int, shape: tuple[int, int]) -> list[np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            masks = archive["masks"]
        if masks.shape != (frame_count, *shape) or masks.dtype != np.uint8:
            return None
        return [mask.copy() for mask in masks]
    except (OSError, ValueError, KeyError):
        # A partial or incompatible cache is treated as a cache miss.
        return None


def _write_cached_masks(path: Path, masks: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(masks).astype(np.uint8, copy=False)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix="masks-", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            np.savez_compressed(temporary_file, masks=stacked)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _grabcut_first_frame(frame: np.ndarray, box: Box) -> np.ndarray:
    """Create a binary foreground mask from a user supplied rectangle."""
    image_height, image_width = frame.shape[:2]
    x, y, box_width, box_height = (int(value) for value in box)
    x0, y0 = max(0, x), max(0, y)
    x1 = min(image_width, x + box_width)
    y1 = min(image_height, y + box_height)
    if box_width <= 0 or box_height <= 0 or x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("The first-frame box must overlap the image by at least 2 pixels.")

    mask = np.full((image_height, image_width), cv2.GC_BGD, dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    # Explicitly mark a small band around the box as background when possible.
    margin = 1
    if x0 > 0:
        mask[y0:y1, max(0, x0 - margin):x0] = cv2.GC_BGD
    if y0 > 0:
        mask[max(0, y0 - margin):y0, x0:x1] = cv2.GC_BGD

    cv2.grabCut(
        frame,
        mask,
        (x0, y0, x1 - x0, y1 - y0),
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if not np.any(binary):
        raise ValueError("GrabCut found no foreground. Try a tighter box around the object.")
    return binary


def _track_mask_affine(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_mask: np.ndarray,
) -> np.ndarray:
    """Track a mask with sparse Lucas-Kanade flow and affine motion."""
    height, width = previous_mask.shape
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        mask=previous_mask,
    )

    affine: np.ndarray | None = None
    if points is not None and len(points) > 0:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            old = points.reshape(-1, 2)[valid]
            new = next_points.reshape(-1, 2)[valid]
            finite = np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
            old, new = old[finite], new[finite]
            if len(old) >= 3:
                affine, inliers = cv2.estimateAffinePartial2D(
                    old,
                    new,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=1000,
                    confidence=0.99,
                )
                if affine is not None and inliers is not None and int(inliers.sum()) < 3:
                    affine = None
            if affine is None and len(old) >= 1:
                # With too few points for rotation/scale, at least follow the
                # median translation so a small mask does not stay behind.
                movement = np.median(new - old, axis=0)
                affine = np.array(
                    [[1.0, 0.0, movement[0]], [0.0, 1.0, movement[1]]],
                    dtype=np.float32,
                )

    if affine is None:
        # No trackable points were found; retain the last known location.
        return previous_mask.copy()

    return cv2.warpAffine(
        previous_mask,
        affine.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def segment_video_mock(
    frames: list[np.ndarray], first_frame_box: Box
) -> list[np.ndarray]:
    """Segment the first frame with GrabCut and follow it through the clip."""
    _validate_frames(frames)
    masks = [_grabcut_first_frame(frames[0], first_frame_box)]
    for index in range(1, len(frames)):
        masks.append(_track_mask_affine(frames[index - 1], frames[index], masks[-1]))
    return masks


def get_video_masks(
    frames: list[np.ndarray],
    object_prompt: str,
    *,
    mock_mode: bool | None = None,
    first_frame_box: Box | None = None,
    cache_dir: str | Path = DEFAULT_MASK_CACHE,
    use_cache: bool = True,
) -> list[np.ndarray]:
    """Return one uint8 binary mask (0 or 255) per frame, using disk cache."""
    if mock_mode is None:
        mock_mode = mock_mode_enabled()
    width, height = _validate_frames(frames)
    if not object_prompt.strip():
        raise ValueError("Describe the object you want to pattern.")
    if mock_mode and first_frame_box is None:
        raise ValueError("Mock segmentation needs a rough box around the object on frame 1.")

    cache_path = _cache_path(
        frames, object_prompt, mock_mode, first_frame_box, cache_dir
    )
    if use_cache:
        cached = _read_cached_masks(cache_path, len(frames), (height, width))
        if cached is not None:
            return cached

    if mock_mode:
        assert first_frame_box is not None
        masks = segment_video_mock(frames, first_frame_box)
    else:
        masks = segment_video_with_sam31(frames, object_prompt)

    if len(masks) != len(frames):
        raise ValueError(
            f"Segmenter returned {len(masks)} masks for {len(frames)} frames."
        )

    normalized_masks: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        array = np.asarray(mask)
        if array.shape != (height, width):
            raise ValueError(
                f"Mask {index} has shape {array.shape}; expected {(height, width)}."
            )
        normalized_masks.append(np.where(array > 0, 255, 0).astype(np.uint8))

    if use_cache:
        _write_cached_masks(cache_path, normalized_masks)
    return normalized_masks
