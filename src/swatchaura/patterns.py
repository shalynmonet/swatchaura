"""Pattern image helpers.

The Monogrammar houndstooth implementation will be wrapped here when its
source is supplied. This module already handles repeating uploaded swatches.
Images returned by these helpers use OpenCV's BGR color order.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

import cv2
import numpy as np

SwatchSource = np.ndarray | bytes | bytearray | memoryview | str | Path

# Ported from Monogrammar's src/pattern/houndstooth.ts. Keep the traced outline
# and 100-unit lattice unchanged so the two-color tile matches that project.
_HOUNDSTOOTH_LATTICE = 100
_HOUNDSTOOTH_MARGIN = 2
_HOUNDSTOOTH_TOOTH: tuple[tuple[float, float], ...] = (
    (113, 26), (117, 27), (93, 51), (93, 74), (45, 123), (44, 96),
    (67, 72), (44, 72), (44, 59), (44, 50), (42, 50), (22, 71),
    (0, 71), (0, 69), (43, 26), (69, 25), (92, 0), (93, 25),
)


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert a six-digit CSS hex color to OpenCV's BGR channel order."""
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"Expected a color in #RRGGBB format, got {color!r}.")
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    return blue, green, red


def generate_houndstooth_tile(
    color1: str = "#F3E3C9",
    color2: str = "#8B3A2B",
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    supersample: int = 4,
) -> np.ndarray:
    """Render Monogrammar's two-color houndstooth as a seamless BGR tile.

    The traced tooth polygon and lattice match Monogrammar's implementation.
    ``scale`` changes the repeat size (and therefore tooth size); ``rotation``
    rotates each tooth around its lattice-cell center while keeping the
    repeating grid aligned, so the result remains tileable at any angle.
    Rasterizing at a larger size and reducing with area interpolation keeps
    diagonal edges smooth instead of pixelated.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pattern scale must be a positive finite number.")
    if not np.isfinite(rotation):
        raise ValueError("Pattern rotation must be a finite number of degrees.")
    if not isinstance(supersample, int) or not 1 <= supersample <= 8:
        raise ValueError("Supersample must be an integer from 1 through 8.")

    background = _hex_to_bgr(color1)
    motif = _hex_to_bgr(color2)
    tile_pixels = max(1, int(round(_HOUNDSTOOTH_LATTICE * scale)))
    pixel_scale = tile_pixels / _HOUNDSTOOTH_LATTICE
    high_width = tile_pixels * supersample
    canvas = np.empty((high_width, high_width, 3), dtype=np.uint8)
    canvas[:] = background

    angle = math.radians(rotation)
    cosine, sine = math.cos(angle), math.sin(angle)
    stroke = max(1, int(round(0.75 * pixel_scale * supersample)))

    # Draw neighboring cells too: a complete tooth extends beyond its own
    # lattice cell, and those overhanging sections form the seamless edges.
    for row in range(-_HOUNDSTOOTH_MARGIN, 1 + _HOUNDSTOOTH_MARGIN):
        for col in range(-_HOUNDSTOOTH_MARGIN, 1 + _HOUNDSTOOTH_MARGIN):
            cell_x = col * _HOUNDSTOOTH_LATTICE
            cell_y = row * _HOUNDSTOOTH_LATTICE
            center_x = cell_x + _HOUNDSTOOTH_LATTICE / 2
            center_y = cell_y + _HOUNDSTOOTH_LATTICE / 2
            points: list[tuple[int, int]] = []
            for point_x, point_y in _HOUNDSTOOTH_TOOTH:
                x = point_x + cell_x - center_x
                y = point_y + cell_y - center_y
                rotated_x = x * cosine - y * sine + center_x
                rotated_y = x * sine + y * cosine + center_y
                points.append(
                    (
                        int(round(rotated_x * pixel_scale * supersample)),
                        int(round(rotated_y * pixel_scale * supersample)),
                    )
                )

            polygon = np.asarray(points, dtype=np.int32)
            cv2.fillPoly(canvas, [polygon], motif, lineType=cv2.LINE_AA)
            # Monogrammar outlines each shape with a 0.75-unit stroke of the
            # same fill color to prevent antialiasing hairlines between shapes.
            cv2.polylines(
                canvas,
                [polygon],
                isClosed=True,
                color=motif,
                thickness=stroke,
                lineType=cv2.LINE_AA,
            )

    if supersample == 1:
        return canvas
    return cv2.resize(
        canvas,
        (tile_pixels, tile_pixels),
        interpolation=cv2.INTER_AREA,
    )


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
