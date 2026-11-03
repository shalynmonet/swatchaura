"""The single adapter for Meta Model API calls.

SAM 3.1 request and response handling follows the official Meta Model API docs
and Meta's ``meta-sam-parser`` client. Keep all future Meta API calls in this
module so the rest of SwatchAura stays independent of API details.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import tempfile
from contextlib import aclosing
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator

import cv2
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import (
    ResponsesEventLike,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI, OpenAI

from .video_io import write_video


BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
MUSE_SPARK_MODEL = "muse-spark-1.3"


def _client() -> AsyncOpenAI:
    """Create the Meta API client using the key stored in the local .env."""
    load_dotenv()
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        raise RuntimeError("Set META_API_KEY in .env before using Meta Model API.")
    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key)


def _sync_client() -> OpenAI:
    """Create a synchronous client for one-shot text model requests."""
    load_dotenv()
    api_key = os.getenv("META_API_KEY")
    if not api_key:
        raise RuntimeError("Set META_API_KEY in .env before using Meta Model API.")
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def request_pattern_settings(style_text: str) -> str:
    """Ask Muse Spark for JSON style settings and return its raw JSON text.

    This is the single Meta API adapter entry point for Muse Spark. The
    endpoint, model, and structured-output request follow Meta's official
    Structured Output guide. The caller validates values before using them.
    """
    prompt = style_text.strip()
    if not prompt:
        raise ValueError("Describe the pattern style you want Muse Spark to parse.")

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "swatchaura_pattern_settings",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "color1": {"type": "string"},
                    "color2": {"type": "string"},
                    "scale": {"type": "number"},
                    "rotation": {"type": "number"},
                    "pattern_type": {
                        "type": "string",
                        "enum": [
                            "houndstooth",
                            "stripes",
                            "polka_dots",
                            "plaid",
                            "floral",
                            "abstract",
                        ],
                    },
                },
                "required": [
                    "color1",
                    "color2",
                    "scale",
                    "rotation",
                    "pattern_type",
                ],
                "additionalProperties": False,
            },
        },
    }

    # TODO: Verify Muse Spark model availability and structured-output access
    # for the user's Meta project/key when live credits are enabled.
    response = _sync_client().chat.completions.create(
        model=MUSE_SPARK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the user's fabric pattern description into the "
                    "requested settings. Use CSS #RRGGBB colors. Scale is a "
                    "positive repeat-size multiplier near 1.0; larger means "
                    "larger motifs. Rotation is degrees from -180 to 180. "
                    "Choose the closest listed pattern_type."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Muse Spark returned an empty style response.")
    return content


def _media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in {"video/mp4", "video/quicktime"}:
        raise ValueError("SAM video uploads must be an MP4 or MOV file.")
    return media_type


async def _normalized_stream_events(
    stream: AsyncIterable[Any],
) -> AsyncIterator[ResponsesEventLike]:
    """Add output-text lane completion events needed by Meta's parser.

    The official API emits ``response.content_part.done`` for text but may not
    emit ``response.output_text.done``. Meta's parser requires the latter to
    close that lane, so synthesize it from the completed content part.
    """
    completed_lanes: set[tuple[object, object, object]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue

        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


async def _upload_and_segment_video(video_path: Path, object_prompt: str) -> Any:
    """Upload one local clip and stream its SAM 3.1 video segmentation."""
    phrase = object_prompt.strip()
    if not phrase:
        raise ValueError("Describe the object you want SAM 3.1 to segment.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file was not found: {video_path}")
    media_type = _media_type(video_path)

    async with _client() as client:
        uploaded = await client.files.create(
            file=(video_path.name, video_path.read_bytes(), media_type),
            purpose="user_data",
        )
        file_id = uploaded.id
        if not isinstance(file_id, str) or not file_id.startswith("file-"):
            raise ValueError("Meta Files API returned an invalid file ID.")

        stream = await client.responses.create(
            model=SAM_MODEL,
            stream=True,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": phrase},
                        {"type": "input_video", "file_id": file_id},
                    ],
                }
            ],
            metadata={"mask_encoding": "one_bit"},
        )

        async with stream, aclosing(_normalized_stream_events(stream)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            async with parsed:
                # Drain the stream so parser.final_result() contains all frames.
                async for _snapshot in parsed:
                    pass
                result = await parsed.final_result()

    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        detail = getattr(result.outcome, "detail", None)
        raise RuntimeError(f"SAM 3.1 segmentation was incomplete: {reason} ({detail})")
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        details = "; ".join(f"line {item.line}: {item.message}" for item in errors)
        raise RuntimeError(f"SAM response parser rejected records: {details}")
    return result


def _result_to_frame_masks(
    result: Any, frames: list[np.ndarray]
) -> list[np.ndarray]:
    """Expand each cropped object mask into a full-frame binary mask.

    The SAM grammar returns a tight mask raster plus a corresponding box. The
    two records are joined by stable object ID and frame index. Multiple
    matches for the noun phrase are combined with a union.
    """
    frame_count = len(frames)
    frame_height, frame_width = frames[0].shape[:2]
    output = [np.zeros((frame_height, frame_width), dtype=np.uint8) for _ in frames]

    def frame_index(record: Any) -> int | None:
        frame_info = record.frame
        return None if frame_info is None else int(frame_info.frame_index)

    boxes = {
        (frame_index(record), record.object_id): record
        for record in result.records
        if record.kind == "box"
    }
    for record in result.records:
        if record.kind != "mask":
            continue
        index = frame_index(record)
        if index is None or not 0 <= index < frame_count:
            continue
        box = boxes.get((index, record.object_id))
        if box is None:
            raise ValueError(
                f"SAM returned a mask without a box for frame {index}, "
                f"object {record.object_id}."
            )

        mask_height = int(record.mask.height)
        mask_width = int(record.mask.width)
        decoded = decode_mask_to_raster(record.mask)
        if isinstance(decoded, (bytes, bytearray, memoryview)):
            raster = np.frombuffer(decoded, dtype=np.uint8)
        else:
            raster = np.asarray(decoded, dtype=np.uint8).reshape(-1)
        if raster.size != mask_height * mask_width:
            raise ValueError("SAM mask dimensions do not match its decoded raster.")
        raster = raster.reshape(mask_height, mask_width)

        # SAM boxes use inclusive right/bottom coordinates. The mask raster is
        # the tight object crop associated with that box.
        box_width = max(1, int(box.right) - int(box.left) + 1)
        box_height = max(1, int(box.bottom) - int(box.top) + 1)
        if raster.shape != (box_height, box_width):
            raster = cv2.resize(
                raster,
                (box_width, box_height),
                interpolation=cv2.INTER_NEAREST,
            )

        left = max(0, int(box.left))
        top = max(0, int(box.top))
        right = min(frame_width, int(box.right) + 1)
        bottom = min(frame_height, int(box.bottom) + 1)
        if left >= right or top >= bottom:
            continue
        crop = raster[
            top - int(box.top): bottom - int(box.top),
            left - int(box.left): right - int(box.left),
        ]
        destination = output[index][top:bottom, left:right]
        np.maximum(destination, crop * np.uint8(255), out=destination)
    return output


def segment_video_with_sam31(
    frames: list[np.ndarray],
    object_prompt: str,
    *,
    video_path: str | Path | None = None,
    fps: float = 24.0,
) -> list[np.ndarray]:
    """Return one full-frame binary SAM 3.1 mask per input frame.

    When ``video_path`` is given, its original clip is uploaded. Otherwise,
    frames are encoded to a temporary MP4 at ``fps`` before upload.
    """
    if not frames:
        raise ValueError("At least one frame is required for SAM segmentation.")

    if video_path is not None:
        path = Path(video_path)
        result = asyncio.run(_upload_and_segment_video(path, object_prompt))
        return _result_to_frame_masks(result, frames)

    with tempfile.TemporaryDirectory(prefix="swatchaura-sam-") as temp_dir:
        path = Path(temp_dir) / "frames.mp4"
        write_video(frames, path, fps)
        result = asyncio.run(_upload_and_segment_video(path, object_prompt))
    return _result_to_frame_masks(result, frames)
