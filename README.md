# SwatchAura

SwatchAura lets you try a print on an object in a real video. Choose what to pattern (for example, an airplane wing or a jacket), describe the style, and render a video preview that follows the object's motion while preserving the scene's lighting and shadows.

The first demo targets are an airplane wing filmed from a window seat and a person walking in a jacket. The fashion workflow helps designers preview prints on garments in motion before ordering fabric samples.

## Project status

This repository is being built milestone by milestone for the Meta Global AI Developer Hackathon (deadline: November 4, 2026). Milestones 1 and 2 (project scaffold and video read/write) are complete. Milestone 3 wraps Monogrammar's houndstooth geometry and adds uploaded swatch tiling. Milestone 4 adds GrabCut/optical-flow mock segmentation, mask caching, and the documented live SAM 3.1 path.

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

## Verify the setup

With the virtual environment active, run:

```powershell
python --version
python -m pip show streamlit opencv-python numpy python-dotenv
```

Python 3.12 works for this project; the minimum supported version is 3.11. The Streamlit application entry point will be added in a later milestone.

## Video I/O (Milestone 2)

`swatchaura.video_io.read_video(path)` reads a clip into OpenCV BGR frames and returns the original frames per second. `swatchaura.video_io.write_video(frames, path, fps)` writes those frames to an MP4 at the provided frame rate. Video resizing and visual effects will be added in later milestones.

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
    cv2.rectangle(frame, (x, 42), (x + 34, 78), (30, 80, 230), thickness=-1)
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

To exercise the live path with an existing clip, switch the local `.env` setting to `MOCK_MODE=false`, then run this in PowerShell. This sends the clip to Meta Model API and uses your API access:

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
6. Optical-flow tracking and homography-based pattern placement, with automatic fallback
7. Validated Muse Spark style settings
8. Muse Image swatch generation
9. Streamlit upload, preview, before/after, and download interface
10. Progress, error handling, sample clips, and final model documentation

## Meta model integration

Live calls are isolated in `meta_client.py` and read `META_API_KEY` from `.env`. SAM 3.1 provides text-prompted object masks, Muse Spark will turn style descriptions into validated pattern settings, and Muse Image will generate swatches. The SAM request follows Meta's [media segmentation guide](https://dev.meta.ai/docs/media-segmentation), [prompt and video upload guide](https://dev.meta.ai/docs/sam/segmenting), and [Python parser guide](https://dev.meta.ai/docs/sam/client-libraries). Muse Spark and Muse Image request formats will be added in their later milestones. Local mock implementations support development without API calls.
