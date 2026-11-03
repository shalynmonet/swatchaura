"""Simple Streamlit interface for the SwatchAura video pipeline."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Make `python -m streamlit run app.py` work directly from a source checkout,
# even when the package has not yet been installed with `pip install -e .`.
SOURCE_DIR = Path(__file__).resolve().parent / "src"
if SOURCE_DIR.is_dir() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from swatchaura.compositing import composite_video
from swatchaura.config import mock_mode_enabled
from swatchaura.patterns import generate_houndstooth_tile
from swatchaura.segmentation import get_video_masks
from swatchaura.style import get_pattern_settings
from swatchaura.swatch import generate_swatch
from swatchaura.video_io import read_video, write_video


MAX_CANVAS_WIDTH = 800
MAX_CANVAS_HEIGHT = 560


def _decode_upload(video_bytes: bytes, suffix: str) -> tuple[list[np.ndarray], float]:
    """Decode uploaded bytes via the project's path-based video helper."""
    with tempfile.TemporaryDirectory(prefix="swatchaura-upload-") as temp_dir:
        video_path = Path(temp_dir) / f"input{suffix}"
        video_path.write_bytes(video_bytes)
        return read_video(video_path)


def _prepare_video(
    video_bytes: bytes, suffix: str
) -> tuple[list[np.ndarray], float]:
    """Cache decoded frames in this browser session for canvas reruns."""
    digest = hashlib.sha256(video_bytes).hexdigest()
    if st.session_state.get("clip_digest") != digest:
        frames, fps = _decode_upload(video_bytes, suffix)
        st.session_state["clip_digest"] = digest
        st.session_state["clip_frames"] = frames
        st.session_state["clip_fps"] = fps
        st.session_state.pop("rendered_video", None)
    return st.session_state["clip_frames"], st.session_state["clip_fps"]


def _resize_frames(
    frames: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    """Resize frames for faster segmentation and preview processing."""
    if scale >= 0.999:
        return frames
    resized = []
    for frame in frames:
        width = max(2, int(round(frame.shape[1] * scale)))
        height = max(2, int(round(frame.shape[0] * scale)))
        resized.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    return resized


def _tighten_masks(
    masks: list[np.ndarray], radius: int
) -> list[np.ndarray]:
    """Shrink coarse mask edges slightly and discard stray boundary pixels."""
    if radius <= 0:
        return masks
    kernel_size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return [cv2.erode(mask, kernel, iterations=1) for mask in masks]


def _mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the selected object so mask quality is easy to inspect."""
    overlay = frame.copy()
    selected = mask > 0
    tint = np.zeros_like(frame)
    tint[:] = (40, 205, 165)
    overlay[selected] = cv2.addWeighted(
        frame[selected], 0.48, tint[selected], 0.52, 0
    )
    contours, _ = cv2.findContours(
        np.asarray(mask > 0, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 220), 1)
    return overlay


def _draw_box(
    first_frame: np.ndarray, canvas_key: str
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """Show the frame and return the last drawn rectangle in frame pixels."""
    frame_height, frame_width = first_frame.shape[:2]
    canvas_scale = min(
        1.0,
        MAX_CANVAS_WIDTH / frame_width,
        MAX_CANVAS_HEIGHT / frame_height,
    )
    canvas_width = max(2, int(round(frame_width * canvas_scale)))
    canvas_height = max(2, int(round(frame_height * canvas_scale)))
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    background = Image.fromarray(rgb).resize(
        (canvas_width, canvas_height), Image.Resampling.LANCZOS
    )
    canvas_result = st_canvas(
        fill_color="rgba(20, 184, 166, 0.22)",
        stroke_color="#14B8A6",
        stroke_width=3,
        background_image=background,
        background_image_fit="stretch",
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="labeled_rect",
        label="object",
        key=canvas_key,
    )
    boxes = canvas_result.boxes or []
    if not boxes:
        return None, None

    box = boxes[-1]
    left = float(box["left"]) * frame_width / canvas_width
    top = float(box["top"]) * frame_height / canvas_height
    width = float(box["width"]) * frame_width / canvas_width
    height = float(box["height"]) * frame_height / canvas_height
    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = int(round(left + width)), int(round(top + height))
    x0 = min(max(0, x0), frame_width - 1)
    y0 = min(max(0, y0), frame_height - 1)
    x1 = min(max(x0 + 1, x1), frame_width)
    y1 = min(max(y0 + 1, y1), frame_height)
    return (x0, y0, x1 - x0, y1 - y0), f"Box: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}"


def _draw_outline(
    first_frame: np.ndarray, canvas_key: str
) -> tuple[np.ndarray | None, str | None]:
    """Let a user trace an exact first-frame silhouette as a polygon."""
    frame_height, frame_width = first_frame.shape[:2]
    canvas_scale = min(
        1.0,
        MAX_CANVAS_WIDTH / frame_width,
        MAX_CANVAS_HEIGHT / frame_height,
    )
    canvas_width = max(2, int(round(frame_width * canvas_scale)))
    canvas_height = max(2, int(round(frame_height * canvas_scale)))
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    background = Image.fromarray(rgb).resize(
        (canvas_width, canvas_height), Image.Resampling.LANCZOS
    )
    canvas_result = st_canvas(
        fill_color="rgba(20, 184, 166, 0.22)",
        stroke_color="#14B8A6",
        stroke_width=3,
        background_image=background,
        background_image_fit="stretch",
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        key=canvas_key,
    )
    drawing = canvas_result.json_data or {}
    polygon = next(
        (
            shape
            for shape in drawing.get("objects", [])
            if str(shape.get("type", "")).lower() == "polygon"
        ),
        None,
    )
    if not polygon:
        return None, None

    offset = polygon.get("pathOffset") or {}
    points = polygon.get("points") or []
    if len(points) < 3:
        return None, None
    scale_x = frame_width / canvas_width
    scale_y = frame_height / canvas_height
    canvas_points = np.asarray(
        [
            [
                (float(point["x"]) + float(offset.get("x", 0))) * scale_x,
                (float(point["y"]) + float(offset.get("y", 0))) * scale_y,
            ]
            for point in points
        ],
        dtype=np.float32,
    )
    vertices = np.round(canvas_points).astype(np.int32)
    outline = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(outline, [vertices], 255)
    return outline, f"Outline contains {int(np.count_nonzero(outline)):,} pixels."


def _make_pattern(
    *,
    style_text: str,
    settings,
    mock_mode: bool,
    uploaded_swatch: bytes | None,
    swatch_prompt: str,
) -> tuple[object, float, float]:
    """Choose an uploaded, generated, or built-in pattern and its scale."""
    if uploaded_swatch:
        return uploaded_swatch, settings.scale, settings.rotation
    if swatch_prompt.strip():
        return (
            generate_swatch(swatch_prompt, mock_mode=mock_mode),
            settings.scale,
            settings.rotation,
        )
    if settings.pattern_type == "houndstooth":
        return (
            generate_houndstooth_tile(settings.color1, settings.color2),
            settings.scale,
            settings.rotation,
        )

    # Other procedural pattern families are available locally, while live
    # mode uses Muse Image to create a swatch for the selected style.
    return generate_swatch(style_text, mock_mode=mock_mode), 1.0, 0.0


def main() -> None:
    st.set_page_config(page_title="SwatchAura", page_icon="🧵", layout="wide")
    st.title("SwatchAura")
    st.caption("Preview a print on a moving object while keeping the original scene lighting.")

    mock_mode = mock_mode_enabled()
    if mock_mode:
        st.info("Mock mode is on. Segmentation and swatch generation run locally without Meta API calls.")
    else:
        st.warning("Live mode is on. Rendering can call SAM 3.1, Muse Spark, or Muse Image and use API access.")

    uploaded_video = st.file_uploader(
        "Upload a short video clip", type=("mp4", "mov"), accept_multiple_files=False
    )
    if uploaded_video is None:
        st.info("Upload an MP4 or MOV clip to begin.")
        st.stop()

    video_bytes = uploaded_video.getvalue()
    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
    try:
        original_frames, fps = _prepare_video(video_bytes, suffix)
    except Exception as exc:
        st.error(f"Could not read this video: {exc}")
        st.stop()

    first = original_frames[0]
    st.caption(
        f"{len(original_frames)} frames · {fps:g} fps · "
        f"{first.shape[1]} × {first.shape[0]} pixels"
    )

    with st.sidebar:
        st.header("Processing")
        automatic_mask_label = (
            "GrabCut from box" if mock_mode else "Automatic (SAM text prompt)"
        )
        mask_method = st.selectbox(
            "Mask method",
            ("Draw precise outline", automatic_mask_label),
            index=0 if mock_mode else 1,
            help=(
                "In mock mode, GrabCut needs a box and does not understand the text prompt. "
                "In live mode, SAM segments the object from the text prompt without a box."
            ),
        )
        resolution_scale = st.select_slider(
            "Processing resolution",
            options=(0.25, 0.5, 0.75, 1.0),
            value=0.5,
            format_func=lambda value: f"{int(value * 100)}%",
            help="Lower resolution speeds up segmentation and preview rendering.",
        )
        export_full_resolution = st.checkbox(
            "Export at original resolution", value=False
        )
        track_surface = st.checkbox("Track print across the surface", value=True)
        mask_tightening = st.slider(
            "Tighten mask edges",
            min_value=0,
            max_value=10,
            value=2,
            help="Erodes the selected area by this many processing pixels. Use a larger value to pull a loose mask inward.",
        )

    processing_frames = _resize_frames(original_frames, resolution_scale)
    processing_height, processing_width = processing_frames[0].shape[:2]
    mock_box = None
    manual_outline = None
    canvas_key = f"object-mask-{st.session_state['clip_digest'][:12]}-{processing_width}x{processing_height}-{mask_method}"
    if mask_method == "Draw precise outline":
        st.subheader("Trace the object on frame 1")
        st.caption("Click the outline corners, then click the first point to close the polygon. The shape is tracked through the clip.")
        manual_outline, outline_caption = _draw_outline(
            processing_frames[0], canvas_key
        )
        if outline_caption:
            st.caption(outline_caption)
        else:
            st.caption("Draw at least three points and close the outline.")
    elif mock_mode and mask_method == "GrabCut from box":
        st.subheader("Mark the object on frame 1")
        st.caption("Draw a snug box around only the surface you want patterned. GrabCut refines its edge and tracks it through the clip; this mock does not interpret the object text prompt.")
        mock_box, box_caption = _draw_box(processing_frames[0], canvas_key)
        if box_caption:
            st.caption(box_caption)
        else:
            st.caption("Click and drag a rectangle around the object to continue.")
    elif mask_method == "Automatic (SAM text prompt)":
        st.subheader("Describe the object")
        st.caption("SAM 3.1 will locate and track it through the uploaded clip.")

    with st.form("render-form"):
        object_prompt = st.text_input(
            "What should be patterned?",
            placeholder="the airplane wing, her jacket, the blue suitcase",
        )
        style_text = st.text_input(
            "Describe the style",
            value="gold and black houndstooth, bigger teeth",
        )
        swatch_prompt = st.text_input(
            "Optional: generate a swatch from a separate prompt",
            placeholder="a repeating indigo and cream floral print",
        )
        swatch_upload = st.file_uploader(
            "Optional: use an existing swatch image",
            type=("png", "jpg", "jpeg", "webp", "bmp"),
            key="swatch-upload",
        )
        render_clicked = st.form_submit_button("Generate preview", type="primary")

    if render_clicked:
        if not object_prompt.strip():
            st.error("Describe which object should receive the pattern.")
        elif mask_method == "Draw precise outline" and manual_outline is None:
            st.error("Trace and close an outline around the object before rendering.")
        elif mock_mode and mask_method == "GrabCut from box" and mock_box is None:
            st.error("Draw a box around the object on frame 1 before rendering.")
        else:
            try:
                progress = st.progress(0, text="Reading style…")
                settings = get_pattern_settings(style_text, mock_mode=mock_mode)
                swatch_bytes = swatch_upload.getvalue() if swatch_upload else None
                pattern, pattern_scale, pattern_rotation = _make_pattern(
                    style_text=style_text,
                    settings=settings,
                    mock_mode=mock_mode,
                    uploaded_swatch=swatch_bytes,
                    swatch_prompt=swatch_prompt,
                )

                progress.progress(15, text="Segmenting the object…")
                # Use the already downscaled frames for live SAM too, so returned
                # mask coordinates match this processing resolution.
                if mask_method == "Draw precise outline":
                    masks = get_video_masks(
                        processing_frames,
                        object_prompt,
                        mock_mode=True,
                        first_frame_mask=manual_outline,
                        video_fps=fps,
                    )
                else:
                    masks = get_video_masks(
                        processing_frames,
                        object_prompt,
                        mock_mode=mock_mode,
                        first_frame_box=mock_box,
                        video_fps=fps,
                    )
                masks = _tighten_masks(masks, mask_tightening)
                if not any(np.any(mask) for mask in masks):
                    raise ValueError(
                        "Mask tightening removed the whole selection. Lower the slider value and render again."
                    )

                if export_full_resolution:
                    output_frames = original_frames
                    output_masks = [
                        cv2.resize(
                            mask,
                            (first.shape[1], first.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for mask in masks
                    ]
                else:
                    output_frames = processing_frames
                    output_masks = masks

                progress.progress(55, text="Applying the print…")
                result_frames = composite_video(
                    output_frames,
                    output_masks,
                    pattern,
                    pattern_scale=pattern_scale,
                    pattern_rotation=pattern_rotation,
                    feather_radius_px=max(1.0, 2.0 * resolution_scale),
                    track_surface=track_surface,
                )

                progress.progress(90, text="Writing the MP4 preview…")
                with tempfile.TemporaryDirectory(prefix="swatchaura-render-") as temp_dir:
                    output_path = write_video(
                        result_frames, Path(temp_dir) / "swatchaura_preview.mp4", fps
                    )
                    result_bytes = output_path.read_bytes()
                st.session_state["rendered_video"] = result_bytes
                st.session_state["rendered_name"] = Path(uploaded_video.name).stem + "_swatchaura.mp4"
                st.session_state["render_settings"] = settings.to_dict()
                st.session_state["mask_overlay"] = cv2.cvtColor(
                    _mask_overlay(output_frames[0], output_masks[0]),
                    cv2.COLOR_BGR2RGB,
                )
                progress.progress(100, text="Preview ready.")
            except Exception as exc:
                st.error(f"Rendering failed: {exc}")
                st.exception(exc)

    result_bytes = st.session_state.get("rendered_video")
    if result_bytes:
        settings = st.session_state.get("render_settings", {})
        st.success(f"Pattern settings: {settings}")
        st.subheader("Preview")
        preview_choice = st.radio(
            "Show",
            ("Before", "Mask", "After"),
            horizontal=True,
            index=2,
            key=f"preview-toggle-{st.session_state['clip_digest'][:12]}",
        )
        if preview_choice == "Mask":
            st.image(
                st.session_state["mask_overlay"],
                caption="First-frame selection: turquoise shows the pixels receiving the pattern.",
                use_container_width=True,
            )
        else:
            st.video(video_bytes if preview_choice == "Before" else result_bytes)
        st.download_button(
            "Download processed video",
            data=result_bytes,
            file_name=st.session_state.get("rendered_name", "swatchaura_preview.mp4"),
            mime="video/mp4",
            type="primary",
        )
        st.caption("This prototype exports video frames without the source audio track.")


if __name__ == "__main__":
    main()
