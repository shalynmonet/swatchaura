# SwatchAura

SwatchAura lets you try a print on an object in a real video. Choose what to pattern (for example, an airplane wing or a jacket), describe the style, and render a video preview that follows the object's motion while preserving the scene's lighting and shadows.

The first demo targets are an airplane wing filmed from a window seat and a person walking in a jacket. The fashion workflow helps designers preview prints on garments in motion before ordering fabric samples.

## Project status

This repository is being built milestone by milestone for the Meta Global AI Developer Hackathon (deadline: November 4, 2026). Milestone 1 (project scaffold and local setup) is complete. Milestone 2 adds video read/write helpers; pattern generation and the app UI will follow later.

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

Live calls will be isolated in `meta_client.py` and will read `META_API_KEY` from `.env`. SAM 3.1 will provide text-prompted object masks, Muse Spark will turn style descriptions into validated pattern settings, and Muse Image will generate swatches. Model endpoint names and request formats will be added from the official API documentation; they are intentionally not guessed in this scaffold. Local mock implementations will make development possible without API credits.
