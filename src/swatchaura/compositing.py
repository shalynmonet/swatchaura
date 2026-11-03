"""Version 1 pattern compositing with lighting and feathered mask edges."""

from __future__ import annotations

import cv2
import numpy as np

from .patterns import SwatchSource, tile_swatch


def _validate_frame_and_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a uint8 BGR color image.")
    if mask.shape != frame.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} must match frame {frame.shape[:2]}.")


def _soft_mask(mask: np.ndarray, feather_radius_px: float) -> np.ndarray:
    """Convert a binary 0/1 or 0/255 mask into a soft 0..1 alpha mask."""
    if not np.isfinite(feather_radius_px) or feather_radius_px < 0:
        raise ValueError("Feather radius must be a non-negative finite number.")

    alpha = np.asarray(mask, dtype=np.float32)
    if alpha.size and float(np.nanmax(alpha)) > 1.0:
        alpha /= 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    if feather_radius_px > 0:
        alpha = cv2.GaussianBlur(
            alpha,
            ksize=(0, 0),
            sigmaX=float(feather_radius_px),
            sigmaY=float(feather_radius_px),
            borderType=cv2.BORDER_REPLICATE,
        )
    return np.clip(alpha, 0.0, 1.0)


def _shade_pattern_with_luminance(
    frame: np.ndarray, tiled_pattern: np.ndarray
) -> np.ndarray:
    """Keep pattern colors while letting source shadows and highlights show."""
    source_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    pattern_lab = cv2.cvtColor(tiled_pattern, cv2.COLOR_BGR2LAB)

    # Neutral light (L=128) leaves the pattern's own lightness unchanged.
    # Dark source pixels shade it down; bright pixels lift it to retain glare.
    source_light = source_lab[:, :, 0].astype(np.float32)
    illumination = 0.35 + 0.65 * (source_light / 128.0)
    pattern_light = pattern_lab[:, :, 0].astype(np.float32)
    pattern_lab[:, :, 0] = np.clip(pattern_light * illumination, 0, 255).astype(
        np.uint8
    )
    return cv2.cvtColor(pattern_lab, cv2.COLOR_LAB2BGR)


def _composite_with_tiled_pattern(
    frame: np.ndarray,
    mask: np.ndarray,
    tiled_pattern: np.ndarray,
    feather_radius_px: float,
) -> np.ndarray:
    _validate_frame_and_mask(frame, mask)
    if tiled_pattern.shape != frame.shape or tiled_pattern.dtype != np.uint8:
        raise ValueError("Tiled pattern must be a uint8 BGR image matching the frame size.")

    alpha = _soft_mask(mask, feather_radius_px)[:, :, None]
    if not np.any(alpha):
        return frame.copy()

    shaded_pattern = _shade_pattern_with_luminance(frame, tiled_pattern)
    blended = (
        frame.astype(np.float32) * (1.0 - alpha)
        + shaded_pattern.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def composite_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    pattern: SwatchSource,
    *,
    pattern_scale: float = 1.0,
    pattern_rotation: float = 0.0,
    feather_radius_px: float = 2.0,
) -> np.ndarray:
    """Fill one object's mask with a repeated, luminance-shaded pattern.

    ``pattern`` can be a generated tile, an uploaded image array, encoded
    upload bytes, or a path. Pattern pixels use OpenCV BGR color order.
    """
    _validate_frame_and_mask(frame, mask)
    height, width = frame.shape[:2]
    tiled = tile_swatch(
        pattern,
        (width, height),
        scale=pattern_scale,
        rotation=pattern_rotation,
    )
    return _composite_with_tiled_pattern(
        frame, mask, tiled, feather_radius_px
    )


def composite_video(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    pattern: SwatchSource,
    *,
    pattern_scale: float = 1.0,
    pattern_rotation: float = 0.0,
    feather_radius_px: float = 2.0,
) -> list[np.ndarray]:
    """Composite the same pattern into each frame using its corresponding mask."""
    if len(frames) != len(masks):
        raise ValueError(
            f"Got {len(frames)} frames and {len(masks)} masks; counts must match."
        )
    if not frames:
        return []

    first = frames[0]
    _validate_frame_and_mask(first, masks[0])
    frame_height, frame_width = first.shape[:2]
    for index, (frame, mask) in enumerate(zip(frames, masks)):
        if frame.shape != first.shape or frame.dtype != np.uint8:
            raise ValueError(f"Frame {index} does not match the first frame's shape/dtype.")
        _validate_frame_and_mask(frame, mask)

    tiled = tile_swatch(
        pattern,
        (frame_width, frame_height),
        scale=pattern_scale,
        rotation=pattern_rotation,
    )
    return [
        _composite_with_tiled_pattern(frame, mask, tiled, feather_radius_px)
        for frame, mask in zip(frames, masks)
    ]
