"""Pattern image helpers.

The Monogrammar houndstooth implementation will be wrapped here when its
source is supplied. This module already handles repeating uploaded swatches.
Images returned by these helpers use OpenCV's BGR color order.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SwatchSource = np.ndarray | bytes | bytearray | memoryview | str | Path


def _read_swatch(source: SwatchSource) -> np.ndarray:
    """Read a BGR swatch from an array, encoded upload bytes, or a path."""
    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Swatch arrays must be grayscale, BGR, or BGRA images.")
        if image.dtype != np.uint8:
            raise ValueError("Swatch arrays must use uint8 pixel values.")
        return np.ascontiguousarray(image)

    if isinstance(source, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError("Could not read the swatch image. Use PNG, JPEG, or BMP.")
    return image


def tile_swatch(
    source: SwatchSource,
    output_size: tuple[int, int],
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
) -> np.ndarray:
    """Repeat a swatch to fill a requested ``(width, height)`` image.

    ``scale`` changes each repeated swatch tile's size (values below 1 make a
    denser print). ``rotation`` turns each tile in place before repeating it.
    This is useful for repeating tileable swatches; arbitrary source edges are
    preserved, so a non-tileable source may have visible joins.
    """
    width, height = (int(value) for value in output_size)
    if width <= 0 or height <= 0:
        raise ValueError("Output width and height must both be positive.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Swatch scale must be a positive finite number.")
    if not np.isfinite(rotation):
        raise ValueError("Swatch rotation must be a finite number of degrees.")

    swatch = _read_swatch(source)
    source_height, source_width = swatch.shape[:2]
    tile_width = max(1, int(round(source_width * scale)))
    tile_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    tile = cv2.resize(swatch, (tile_width, tile_height), interpolation=interpolation)

    if rotation % 360:
        center = (tile_width / 2.0, tile_height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
        tile = cv2.warpAffine(
            tile,
            matrix,
            (tile_width, tile_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    repeats_x = (width + tile_width - 1) // tile_width
    repeats_y = (height + tile_height - 1) // tile_height
    repeated = np.tile(tile, (repeats_y, repeats_x, 1))
    return repeated[:height, :width].copy()
