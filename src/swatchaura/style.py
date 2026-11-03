"""Rule-based or Muse Spark parsing for fabric pattern descriptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import re
from typing import Any, Mapping

from .config import mock_mode_enabled
from .meta_client import request_pattern_settings

logger = logging.getLogger(__name__)

PATTERN_TYPES = (
    "houndstooth",
    "stripes",
    "polka_dots",
    "plaid",
    "floral",
    "abstract",
)
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class PatternSettings:
    """Validated settings understood by SwatchAura's pattern pipeline."""

    color1: str = "#F3E3C9"
    color2: str = "#8B3A2B"
    scale: float = 1.0
    rotation: float = 0.0
    pattern_type: str = "houndstooth"

    def to_dict(self) -> dict[str, str | float]:
        """Return JSON-friendly values for the UI or pipeline."""
        return asdict(self)


DEFAULT_PATTERN_SETTINGS = PatternSettings()

_COLOR_WORDS: dict[str, str] = {
    "black": "#111111",
    "white": "#F7F5F0",
    "cream": "#F3E3C9",
    "ivory": "#FFF8E7",
    "beige": "#D8C3A5",
    "brown": "#6B442E",
    "rust": "#8B3A2B",
    "red": "#C62828",
    "orange": "#E87524",
    "yellow": "#F2C94C",
    "gold": "#D4AF37",
    "green": "#3D7048",
    "blue": "#315A9B",
    "navy": "#172B4D",
    "purple": "#704080",
    "pink": "#D87991",
    "gray": "#808080",
    "grey": "#808080",
}
_COLOR_TOKEN = re.compile(
    r"#[0-9a-fA-F]{6}|\b(?:" + "|".join(_COLOR_WORDS) + r")\b",
    re.IGNORECASE,
)


def validate_pattern_settings(value: Mapping[str, Any]) -> PatternSettings:
    """Validate model or mock settings before returning them to the pipeline."""
    if not isinstance(value, Mapping):
        raise ValueError("Pattern settings must be a JSON object.")

    color1 = value.get("color1")
    color2 = value.get("color2")
    if not isinstance(color1, str) or not _HEX_COLOR.fullmatch(color1):
        raise ValueError("color1 must be a six-digit #RRGGBB color.")
    if not isinstance(color2, str) or not _HEX_COLOR.fullmatch(color2):
        raise ValueError("color2 must be a six-digit #RRGGBB color.")

    scale = value.get("scale")
    rotation = value.get("rotation")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale must be a number.")
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)):
        raise ValueError("rotation must be a number.")
    scale, rotation = float(scale), float(rotation)
    if not math.isfinite(scale) or not 0.1 <= scale <= 8.0:
        raise ValueError("scale must be between 0.1 and 8.0.")
    if not math.isfinite(rotation) or not -180.0 <= rotation <= 180.0:
        raise ValueError("rotation must be between -180 and 180 degrees.")

    pattern_type = value.get("pattern_type")
    if pattern_type not in PATTERN_TYPES:
        raise ValueError(f"pattern_type must be one of {', '.join(PATTERN_TYPES)}.")

    return PatternSettings(
        color1=color1.upper(),
        color2=color2.upper(),
        scale=scale,
        rotation=rotation,
        pattern_type=pattern_type,
    )


def parse_style_mock(style_text: str) -> PatternSettings:
    """Extract common colors, size words, rotation, and pattern type locally."""
    text = style_text.strip().lower()
    if not text:
        return DEFAULT_PATTERN_SETTINGS

    colors: list[str] = []
    for match in _COLOR_TOKEN.finditer(text):
        token = match.group(0)
        colors.append(
            token.upper() if token.startswith("#") else _COLOR_WORDS[token.lower()]
        )
        if len(colors) == 2:
            break
    color1 = colors[0] if colors else DEFAULT_PATTERN_SETTINGS.color1
    color2 = colors[1] if len(colors) > 1 else DEFAULT_PATTERN_SETTINGS.color2

    scale = 1.0
    if re.search(r"\b(tiny|micro|much smaller|smaller|fine)\b", text):
        scale = 0.55
    if re.search(r"\b(large|larger|bigger|oversized|much bigger)\b", text):
        scale = 1.6
    if re.search(r"\b(huge|giant)\b", text):
        scale = 2.2

    rotation = 0.0
    rotation_match = re.search(
        r"\b(?:rotate|rotation|turn)\s*(-?\d+(?:\.\d+)?)\s*(?:degrees?|°)?",
        text,
    )
    if rotation_match:
        rotation = float(rotation_match.group(1))

    pattern_type = "houndstooth"
    for phrase, kind in (
        ("polka dot", "polka_dots"),
        ("polka-dot", "polka_dots"),
        ("stripe", "stripes"),
        ("plaid", "plaid"),
        ("floral", "floral"),
        ("abstract", "abstract"),
    ):
        if phrase in text:
            pattern_type = kind
            break

    return validate_pattern_settings(
        {
            "color1": color1,
            "color2": color2,
            "scale": scale,
            "rotation": rotation,
            "pattern_type": pattern_type,
        }
    )


def get_pattern_settings(
    style_text: str, *, mock_mode: bool | None = None
) -> PatternSettings:
    """Return validated pattern settings from mock rules or Muse Spark.

    Invalid or malformed model JSON is logged and replaced with safe defaults.
    API/network errors still raise so the app can show an actionable message.
    """
    if mock_mode is None:
        mock_mode = mock_mode_enabled()
    if mock_mode:
        return parse_style_mock(style_text)

    raw = request_pattern_settings(style_text)
    try:
        decoded = json.loads(raw)
        return validate_pattern_settings(decoded)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Muse Spark returned invalid pattern settings; defaults used: %s", exc)
        return DEFAULT_PATTERN_SETTINGS
