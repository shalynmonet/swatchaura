# SwatchAura

SwatchAura lets you try a print on an object in a real video. Choose what to pattern (for example, an airplane wing or a jacket), describe the style, and render a video preview that follows the object's motion while preserving the scene's lighting and shadows.

The first demo targets are an airplane wing filmed from a window seat and a person walking in a jacket. The fashion workflow helps designers preview prints on garments in motion before ordering fabric samples.

## Project status

This repository is being built milestone by milestone for the Meta Global AI Developer Hackathon (deadline: November 4, 2026). Milestones 1 and 2 (project scaffold and video read/write) are complete. Milestone 3 wraps Monogrammar's houndstooth geometry and adds uploaded swatch tiling. Milestone 4 adds GrabCut/optical-flow mock segmentation, mask caching, and the documented live SAM 3.1 path. Milestone 5 composites patterns into the masks while keeping source lighting and soft edges. Milestone 6 combines homography placement with dense optical-flow deformation for local surface movement and automatic fallback. Milestone 7 parses style descriptions into validated pattern settings using local rules or Muse Spark. Milestone 8 generates swatches locally or with Muse Image and passes them to the same video compositor. Milestone 9 provides the Streamlit upload, mask drawing, preview, and download flow.

## Requirements

- Windows with PowerShell
- Python 3.11 or newer
- Git (optional, for version control)

## Set up in PowerShell

Run these commands from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

The Streamlit entry point also adds the repository's `src` folder to Python's import path, so it can load the `swatchaura` package directly from a checkout. If you see `ModuleNotFoundError: No module named 'swatchaura'`, confirm PowerShell is in the project folder and run `python -m pip install -e .` in the active virtual environment.

If PowerShell blocks virtual environment activation, run this for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The `.env` file is ignored by Git. Keep your real API key there; never put it in source code or commit it. The example starts in mock mode and has no key configured:

```dotenv
MOCK_MODE=true
META_API_KEY=
```

`MOCK_MODE=true` uses local implementations for segmentation and style parsing, and requires no Meta API key. Set it to `false` to use live Meta Model API calls.

## Verify the setup

With the virtual environment active, run:

```powershell
python --version
python -m pip show streamlit opencv-python numpy python-dotenv
```

Python 3.12 works for this project; the minimum supported version is 3.11.

## Run the Streamlit app (Milestone 9)

With the virtual environment active and dependencies installed, start the UI from the project folder:

```powershell
python -m streamlit run app.py
```

Upload an MP4 or MOV, describe the object, enter the style, and click **Generate preview**. With `MOCK_MODE=true`, **Draw precise outline** is the default: click the silhouette vertices on frame one, then click the first point to close it. The traced mask is tracked through the clip and cached. You can also choose **GrabCut from box** for a quick rough selection; GrabCut does not understand the object text prompt. Set `MOCK_MODE=false` and choose **Automatic (SAM text prompt)** to let SAM 3.1 find the object from a description such as “the airplane wing” or “her jacket” without drawing a mask. Choose **Draw precise outline** in live mode to use a hand-drawn mask instead. The **Mask** preview shows the selected first-frame area, and **Tighten mask edges** shrinks the selection if it includes too much background. Local rules generate styles in mock mode. You can upload a swatch image or enter a separate swatch prompt. In live mode, Muse Spark parses the style and a swatch prompt invokes Muse Image. Select **Before**, **Mask**, or **After**, then download the rendered MP4.

The sidebar processes a smaller video by default for speed and can export the result at the source video's resolution. The prototype preserves frame rate but does not retain source audio. The drawing control is provided by `streamlit-drawable-canvas` 0.13, which requires Streamlit 1.53 or newer. The MP4 writer uses H.264 baseline encoding for browser playback.

## Video I/O (Milestone 2)

`swatchaura.video_io.read_video(path)` reads a clip into OpenCV BGR frames and returns the original frames per second. `swatchaura.video_io.write_video(frames, path, fps)` writes those frames to an H.264 MP4 at the provided frame rate, for browser playback. `imageio-ffmpeg` supplies the encoder on Windows, macOS, and Linux. Video resizing and visual effects will be added in later milestones.

To quickly check the read/write helpers, activate the virtual environment and run this PowerShell here-string. It makes a tiny synthetic clip, reads it, writes a copy, then prints the frame count and frame rate:

```powershell
@'
import cv2
import numpy as np
from swatchaura.video_io import read_video, write_video

source = "sample_input.mp4"
frames = [np.full((120, 160, 3), (30, 120, 220), dtype=np.uint8) for _ in range(12)]
write_video(frames, source, 12)
read_frames, fps = read_video(source)
write_video(read_frames, "sample_output.mp4", fps)
print(f"Read {len(read_frames)} frames at {fps:g} fps; wrote sample_output.mp4")
'@ | python -
```

## Segmentation (Milestone 4)

`swatchaura.segmentation.get_video_masks(...)` returns one binary mask per video frame. With `MOCK_MODE=true` (the default), give it a rough first-frame box as `(x, y, width, height)`. OpenCV GrabCut creates the first mask, then sparse optical flow follows that mask through subsequent frames. The helper caches the masks in `cache/masks/`; the cache key includes the clip contents, object prompt, mode, and box. A changed clip or box automatically gets a separate cache entry.

To exercise the mock without a sample clip, run this PowerShell here-string from the project folder with the virtual environment active. It creates a small moving-shape clip, segments it, and prints mask and cache information:

```powershell
@'
import cv2
import numpy as np
from swatchaura.segmentation import get_video_masks
from swatchaura.video_io import read_video, write_video

frames = []
for index in range(12):
    frame = np.full((120, 160, 3), (45, 45, 45), dtype=np.uint8)
    x = 62 + index * 2
    cv2.rectangle(frame, (x, 42), (x + 34, 78), (230, 80, 30), thickness=-1)
    frames.append(frame)

write_video(frames, "sample_segmentation.mp4", 12)
clip_frames, _ = read_video("sample_segmentation.mp4")
masks = get_video_masks(
    clip_frames,
    "the moving blue rectangle",
    mock_mode=True,
    first_frame_box=(57, 37, 52, 48),
)
print(f"Got {len(masks)} masks with shape {masks[0].shape}.")
print(f"Foreground pixels in frame 1: {np.count_nonzero(masks[0])}")
print(f"Cached mask files: {len(list(__import__('pathlib').Path('cache/masks').glob('*.npz')))}")
'@ | python -
```

The live SAM 3.1 call is isolated in `swatchaura.meta_client` and follows Meta's documented `/v1/files` upload and streaming `/v1/responses` flow, parsed with `meta-sam-parser`. It reads the project's `META_API_KEY` from `.env`. In live mode, the source clip is uploaded to Meta Model API with `purpose="user_data"`. The first-frame box can be supplied from a UI drawing control when the Streamlit milestone is reached.

First run the mock example above once; it creates `sample_segmentation.mp4` with a blue rectangle. Then switch the local `.env` setting to `MOCK_MODE=false` and run this in PowerShell. This sends the clip to Meta Model API and uses your API access. Replace the path with your own MP4 or MOV to test another clip.

```powershell
@'
from pathlib import Path
from swatchaura.segmentation import get_video_masks
from swatchaura.video_io import read_video

video_path = Path("sample_segmentation.mp4")
frames, fps = read_video(video_path)
masks = get_video_masks(
    frames,
    "the blue rectangle",
    video_path=video_path,
    video_fps=fps,
)
print(f"SAM returned {len(masks)} frame masks; frame 1 has {int((masks[0] > 0).sum())} foreground pixels.")
'@ | python -
```

For offline development, keep `MOCK_MODE=true` (or pass `mock_mode=True`) and provide `first_frame_box` as shown in the mock example above.

## Version 1 compositing (Milestone 5)

`swatchaura.compositing.composite_frame(...)` and `composite_video(...)` tile a generated pattern or uploaded swatch into each object mask. The pattern lightness is shaded by the source frame's LAB luminance so folds, shadows, and highlights show through. A Gaussian feather softens mask edges. This remains the reliable image-space fallback when surface tracking is unavailable.

After running the mock segmentation example above once, use this PowerShell block to make a patterned video preview. It uses only local mock segmentation:

```powershell
@'
import cv2
import numpy as np
from pathlib import Path
from swatchaura.compositing import composite_video
from swatchaura.patterns import generate_houndstooth_tile
from swatchaura.segmentation import get_video_masks
from swatchaura.video_io import read_video, write_video

video_path = Path("sample_segmentation.mp4")
# Recreate the known test clip so an older file with different box coordinates
# cannot make this example composite the wrong part of the frame.
frames = []
for index in range(12):
    frame = np.full((120, 160, 3), (45, 45, 45), dtype=np.uint8)
    x = 62 + index * 2
    cv2.rectangle(frame, (x, 42), (x + 34, 78), (230, 80, 30), thickness=-1)
    frames.append(frame)
write_video(frames, video_path, 12)
frames, fps = read_video(video_path)
masks = get_video_masks(
    frames,
    "blue rectangle",
    mock_mode=True,
    first_frame_box=(57, 37, 52, 48),
    video_path=video_path,
    video_fps=fps,
)
pattern = generate_houndstooth_tile("#F3E3C9", "#8B3A2B")
result = composite_video(frames, masks, pattern, pattern_scale=1.0, feather_radius_px=2.0)
write_video(result, "sample_composited.mp4", fps)
print(f"Wrote {len(result)} patterned frames at {fps:g} fps to sample_composited.mp4")
'@ | python -
```

Open `sample_composited.mp4` to inspect the before/after effect. The generated sample output is ignored by Git.

## Surface tracking (Milestone 6)

Pass `track_surface=True` to `composite_video(...)` to track Shi-Tomasi feature points inside the first-frame mask with pyramidal Lucas-Kanade optical flow. RANSAC estimates a global homography, and dense forward/backward-checked optical flow locally deforms the pattern between adjacent frames. This lets the print respond to visible bends and non-rigid motion instead of moving as one flat plane. When local flow is unreliable, the global homography is used; when both tracking methods fail, the compositor uses version 1 for that frame. Low-texture or stationary regions may not provide enough visual information for local deformation.

This example makes a small moving textured patch so optical flow has visual features to follow. With the virtual environment active, run:

```powershell
@'
import numpy as np
from swatchaura.compositing import composite_video
from swatchaura.patterns import generate_houndstooth_tile
from swatchaura.video_io import write_video

rng = np.random.default_rng(9)
patch = rng.integers(35, 220, size=(48, 56, 3), dtype=np.uint8)
frames = []
masks = []
for index in range(16):
    frame = np.full((140, 180, 3), 35, dtype=np.uint8)
    x, y = 55 + index * 2, 42 + index
    frame[y:y + 48, x:x + 56] = patch
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y + 48, x:x + 56] = 255
    frames.append(frame)
    masks.append(mask)

pattern = generate_houndstooth_tile("#F3E3C9", "#8B3A2B")
result = composite_video(
    frames,
    masks,
    pattern,
    track_surface=True,
    feather_radius_px=1.5,
)
write_video(result, "sample_tracked.mp4", 12)
print("Wrote sample_tracked.mp4; the print should move with the textured patch.")
'@ | python -
Start-Process .\sample_tracked.mp4
```

## Style parsing (Milestone 7)

`swatchaura.style.get_pattern_settings(...)` converts a description to validated `color1`, `color2` (hex), `scale`, `rotation`, and `pattern_type` values. In mock mode it uses a small local ruleset for color names/hex codes, size words, rotation degrees, and common pattern types. In live mode it asks Muse Spark `muse-spark-1.3` for schema-constrained JSON through the single `meta_client.py` adapter. Malformed JSON or out-of-range values log a warning and return the default settings; API/network errors remain visible to the app. Pattern type is parsed as metadata; the current built-in renderer still draws houndstooth, while other pattern types can be paired with swatches in a later step.

With the virtual environment active and `MOCK_MODE=true`, this local example costs no API usage:

```powershell
@'
from swatchaura.style import get_pattern_settings

settings = get_pattern_settings("gold and black houndstooth, bigger teeth")
print(settings.to_dict())
'@ | python -
```

To try Muse Spark, set `MOCK_MODE=false` in `.env`, make sure `META_API_KEY` is set, then run the same block. That makes a live API request. The request uses Meta's official [Muse Spark structured-output guide](https://dev.meta.ai/docs/structured-output), including JSON Schema output via Chat Completions. API request details remain isolated in `src/swatchaura/meta_client.py`.

## Swatch generation (Milestone 8)

`swatchaura.swatch.generate_swatch(prompt)` returns an OpenCV BGR image tile that can go directly into `composite_video(...)`. With `MOCK_MODE=true`, a local procedural renderer makes houndstooth, stripes, polka dots, plaid, floral, or abstract tiles. With `MOCK_MODE=false`, the prompt goes to Muse Image `muse-image-1.0` through `meta_client.py`; the adapter decodes its base64 image response to bytes and the swatch module turns those bytes into an image array.

This local example makes a patterned preview without Meta API usage. It creates a textured moving patch and matching masks so the surface tracking option has features to follow:

```powershell
@'
import numpy as np
from swatchaura.compositing import composite_video
from swatchaura.swatch import generate_swatch
from swatchaura.video_io import write_video

prompt = "gold and black houndstooth, bigger teeth"
swatch = generate_swatch(prompt, mock_mode=True)
rng = np.random.default_rng(9)
patch = rng.integers(35, 220, size=(48, 56, 3), dtype=np.uint8)
frames, masks = [], []
for index in range(16):
    frame = np.full((140, 180, 3), 35, dtype=np.uint8)
    x, y = 55 + index * 2, 42 + index
    frame[y:y + 48, x:x + 56] = patch
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y + 48, x:x + 56] = 255
    frames.append(frame)
    masks.append(mask)

result = composite_video(
    frames, masks, swatch, track_surface=True, feather_radius_px=1.5
)
write_video(result, "sample_muse_swatch.mp4", 12)
print("Wrote sample_muse_swatch.mp4 using the generated swatch.")
'@ | python -
Start-Process .\sample_muse_swatch.mp4
```

For live generation, set `MOCK_MODE=false` in `.env`, ensure `META_API_KEY` is set, and call `generate_swatch(prompt)` without a `mock_mode` override. That sends one generation request to Meta. The request follows Meta's [Muse Image generation guide](https://dev.meta.ai/docs/image-generation) and [Generate an image API reference](https://dev.meta.ai/docs/api-reference/images/create-image). Model API request formats and credentials stay inside `meta_client.py`.

## Uploaded swatch tiling (Milestone 3)

`swatchaura.patterns.tile_swatch(source, output_size, scale=..., rotation=...)` repeats an image to fill `(width, height)`. It accepts an OpenCV image array, an image path, or encoded upload bytes (for example, a Streamlit upload's `getvalue()`). `scale` sets the size of each repeat and `rotation` turns each repeat. A non-tileable source can have visible joins; uploaded tileable swatches repeat cleanly.

## Houndstooth pattern (Milestone 3)

`swatchaura.patterns.generate_houndstooth_tile(color1, color2, scale=..., rotation=...)` ports the traced tooth polygon and 100-unit repeat lattice from Monogrammar's `src/pattern/houndstooth.ts`. It renders the two hex colors to a compact seamless BGR image, using supersampling for smooth diagonal edges. Scale changes the tooth and repeat size. Rotation turns each tooth inside its repeat cell so the output still tiles cleanly.

To create two preview images and open them in Windows, run this from the project folder with the virtual environment active:

```powershell
@'
import cv2
from swatchaura.patterns import generate_houndstooth_tile, tile_swatch

tile = generate_houndstooth_tile(
    color1="#F3E3C9",
    color2="#8B3A2B",
    scale=1.0,
    rotation=0,
)
cv2.imwrite("houndstooth_tile.png", tile)
preview = tile_swatch(tile, (600, 400), scale=1.0)
cv2.imwrite("houndstooth_preview.png", preview)
print(f"Tile: {tile.shape[1]} x {tile.shape[0]}; preview saved.")
'@ | python -
Start-Process .\houndstooth_preview.png
```

## Planned build

1. Project scaffold and local setup
2. Video read/write while preserving the clip's frame rate
3. Crisp, tileable houndstooth and arbitrary swatch tiling
4. Per-frame segmentation, starting with a local mock, plus disk caching
5. Reliable version 1 compositing with luminance, shadows, and feathered edges
6. Optical-flow tracking and homography-based pattern placement, with automatic fallback (implemented)
7. Validated Muse Spark style settings (implemented)
8. Muse Image swatch generation (implemented)
9. Streamlit upload, preview, before/after, and download interface (implemented)
10. Progress, error handling, sample clips, and final model documentation

## Meta model integration

Live calls are isolated in `meta_client.py` and read `META_API_KEY` from `.env`. SAM 3.1 provides text-prompted object masks, Muse Spark will turn style descriptions into validated pattern settings, and Muse Image will generate swatches. The SAM request follows Meta's [media segmentation guide](https://dev.meta.ai/docs/media-segmentation), [prompt and video upload guide](https://dev.meta.ai/docs/sam/segmenting), and [Python parser guide](https://dev.meta.ai/docs/sam/client-libraries). Muse Spark and Muse Image request formats will be added in their later milestones. Local mock implementations support development without API calls.
