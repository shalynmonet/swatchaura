"""The single adapter for Meta Model API calls.

Endpoint names and request formats are intentionally left out until the Meta
Model API documentation is available. Keep all future Meta HTTP calls in this
module so the rest of SwatchAura stays independent of the API details.
"""

from __future__ import annotations

import numpy as np


def segment_video_with_sam31(
    frames: list[np.ndarray], object_prompt: str
) -> list[np.ndarray]:
    """TODO: call SAM 3.1 and return one binary mask per input frame.

    Fill in the endpoint, authentication header, upload flow, and response
    parsing only after the official Meta Model API docs are provided. Read
    ``META_API_KEY`` from the environment here; never embed a key in source.
    """
    raise NotImplementedError(
        "SAM 3.1 API wiring is waiting for the official Meta Model API docs. "
        "Set MOCK_MODE=true to use local GrabCut and tracking."
    )
