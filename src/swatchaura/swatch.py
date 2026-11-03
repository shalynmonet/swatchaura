"""Generate reusable swatches locally or with Muse Image."""

from __future__ import annotations

import cv2
import numpy as np

from .config import mock_mode_enabled
from .meta_client import request_pattern_swatch
from .patterns import generate_houndstooth_tile
from .style import get_pattern_settings


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert a validated CSS hex color to OpenCV channel order."""
    return (int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16))


def _period_for_scale(scale: float, size: int) -> int:
    """Choose a repeat period that divides the square tile exactly."""
    candidates = [period for period in (16, 32, 64, 128) if size % period == 0]
    target = 64.0 * scale
    return min(candidates, key=lambda period: abs(period - target))


def generate_procedural_swatch(
    prompt: str, *, size: int = 256
) -> np.ndarray:
    """Make a simple tileable mock swatch from a prompt, without API credits."""
    if not isinstance(size, int) or size < 32 or size % 16:
        raise ValueError("Swatch size must be an integer of at least 32 divisible by 16.")
    settings = get_pattern_settings(prompt, mock_mode=True)
    foreground = _hex_to_bgr(settings.color1)
    accent = _hex_to_bgr(settings.color2)
    period = _period_for_scale(settings.scale, size)
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:] = foreground
    yy, xx = np.indices((size, size))

    if settings.pattern_type == "houndstooth":
        return generate_houndstooth_tile(
            settings.color1,
            settings.color2,
            scale=max(0.25, min(2.0, settings.scale)),
            rotation=settings.rotation,
        )

    if settings.pattern_type == "stripes":
        stripe_width = max(2, period // 4)
        stripes = ((xx + yy) % period) < stripe_width
        tile[stripes] = accent
    elif settings.pattern_type == "plaid":
        stripe_width = max(2, period // 5)
        vertical = (xx % period) < stripe_width
        horizontal = (yy % period) < stripe_width
        tile[vertical | horizontal] = accent
        tile[vertical & horizontal] = tuple((c + b) // 2 for c, b in zip(foreground, accent))
    elif settings.pattern_type == "polka_dots":
        radius = max(3, period // 6)
        for center_y in range(-period, size + period, period):
            row = center_y // period
            offset = period // 2 if row % 2 else 0
            for center_x in range(-period + offset, size + period, period):
                cv2.circle(
                    tile,
                    (center_x, center_y),
                    radius,
                    accent,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
    elif settings.pattern_type == "floral":
        radius = max(4, period // 6)
        petal_centers = ((-radius, 0), (radius, 0), (0, -radius), (0, radius))
        for center_y in range(-period, size + period, period):
            for center_x in range(-period, size + period, period):
                for dx, dy in petal_centers:
                    cv2.circle(
                        tile,
                        (center_x + dx, center_y + dy),
                        radius,
                        accent,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
                cv2.circle(
                    tile,
                    (center_x, center_y),
                    max(2, radius // 2),
                    foreground,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
    else:  # abstract: a crisp alternating repeat for the local placeholder.
        diamonds = ((xx % period) + (yy % period)) % period < period // 2
        tile[diamonds] = accent

    if settings.rotation % 360:
        matrix = cv2.getRotationMatrix2D(
            (size / 2.0, size / 2.0), settings.rotation, 1.0
        )
        tile = cv2.warpAffine(
            tile,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
    return tile


def generate_swatch(
    prompt: str, *, mock_mode: bool | None = None
) -> np.ndarray:
    """Return a BGR pattern tile using local mock art or Meta Muse Image.

    The resulting array can be passed directly as ``pattern`` to
    ``composite_frame`` or ``composite_video``.
    """
    if not prompt.strip():
        raise ValueError("Describe the swatch pattern you want to create.")
    if mock_mode is None:
        mock_mode = mock_mode_enabled()
    if mock_mode:
        return generate_procedural_swatch(prompt)

    image_bytes = request_pattern_swatch(prompt)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Muse Image returned data that is not a readable image.")
    return image
