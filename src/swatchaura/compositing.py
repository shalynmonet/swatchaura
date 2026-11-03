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
    track_surface: bool = False,
) -> list[np.ndarray]:
    """Composite a pattern into each mask, optionally tracking the surface.

    With ``track_surface=True``, feature points inside the first mask are
    tracked through the clip. A homography maps the first-frame pattern into
    each later frame. If a frame has too few reliable tracks, it uses the
    ordinary image-space composite for that frame instead.
    """
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
    if not track_surface:
        return [
            _composite_with_tiled_pattern(frame, mask, tiled, feather_radius_px)
            for frame, mask in zip(frames, masks)
        ]

    homographies = _track_pattern_homographies(frames, masks[0])
    output: list[np.ndarray] = []
    for frame, mask, homography in zip(frames, masks, homographies):
        if homography is None:
            output.append(
                _composite_with_tiled_pattern(frame, mask, tiled, feather_radius_px)
            )
            continue

        warped_pattern = cv2.warpPerspective(
            tiled,
            homography,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        output.append(
            _composite_with_tiled_pattern(
                frame, mask, warped_pattern, feather_radius_px
            )
        )
    return output


def _track_pattern_homographies(
    frames: list[np.ndarray], first_mask: np.ndarray
) -> list[np.ndarray | None]:
    """Track first-frame features and estimate first-to-current homographies.

    ``None`` marks a frame where optical flow or the robust homography fit
    failed; the caller should use the version 1 image-space composite there.
    """
    height, width = frames[0].shape[:2]
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    feature_mask = np.where(first_mask > 0, 255, 0).astype(np.uint8)
    reference_points = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        mask=feature_mask,
    )
    homographies: list[np.ndarray | None] = [np.eye(3, dtype=np.float64)]
    if reference_points is None or len(reference_points) < 4:
        return homographies + [None] * (len(frames) - 1)

    reference_points = reference_points.reshape(-1, 2).astype(np.float32)
    tracked_points = reference_points.copy()
    previous_gray = first_gray
    for frame in frames[1:]:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            tracked_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        homography: np.ndarray | None = None
        if next_points is not None and status is not None:
            next_flat = next_points.reshape(-1, 2)
            valid = status.reshape(-1).astype(bool)
            valid &= np.isfinite(next_flat).all(axis=1)
            valid &= (next_flat[:, 0] >= 0) & (next_flat[:, 0] < width)
            valid &= (next_flat[:, 1] >= 0) & (next_flat[:, 1] < height)
            source = reference_points[valid]
            target = next_flat[valid]
            if len(source) >= 4:
                candidate, inliers = cv2.findHomography(
                    source, target, cv2.RANSAC, 3.0
                )
                if candidate is not None and inliers is not None:
                    inlier_count = int(inliers.sum())
                    inlier_ratio = inlier_count / len(source)
                    if (
                        inlier_count >= 6
                        and inlier_ratio >= 0.35
                        and np.isfinite(candidate).all()
                    ):
                        homography = candidate
            # Carry only successfully tracked points into the next flow step,
            # while retaining their matching frame-zero reference coordinates.
            reference_points = reference_points[valid]
            tracked_points = next_flat[valid]
        else:
            homographies.append(None)
            homographies.extend([None] * (len(frames) - len(homographies)))
            break
        homographies.append(homography)
        previous_gray = current_gray
        if len(tracked_points) < 4:
            homographies.extend([None] * (len(frames) - len(homographies)))
            break
    return homographies
