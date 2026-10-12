# Next-chat handoff

Last updated: 2026-09-29

## User’s intended result

The user wants a real personalized video virtual try-on. When a person records a whole moving video, the system should use the full temporal sequence—not one frame—to learn visual body/pose/occlusion context and later render the exact suit as if it is being worn. The clipboard-like PNG overlay is not acceptable as the final result.

Do not describe this as body measurement, sizing, medical profiling, or guaranteed physical fit. The allowed target is visual appearance and movement continuity.

## What was implemented

- `prototype/app.js`:
  - local camera preview;
  - temporary upright pose-guided garment guide only as a clearly labeled fallback;
  - automatic fitted-still request after camera permission;
  - explicit full-video recorder using `MediaRecorder`;
  - calibration stages `reference_neutral_clothing` and `wearing_exact_garment`;
  - observed frame timing metadata via `requestVideoFrameCallback` when available, with a timed fallback;
  - explicit **Start full-video recording** and **Stop & save full recording** controls.
- `prototype/local_server.py`:
  - loopback-only static server;
  - `/api/tryon` endpoint for local FASHN still-image inference;
  - request images remain in memory and are not uploaded or saved by the server.
- The recorder downloads a complete `.webm` plus a `.json` sidecar containing capture stage, times, dimensions, frame-timing entries, and SHA-256.

## Current truth

- Camera-on alone does not save frames.
- No real calibration video has been confirmed in Downloads or the repository.
- FASHN model and DWPose weights are present under ignored local storage.
- CUDA is available for private research/training and local evaluation. The final API/frontend path must not depend on server CUDA; it should use a validated portable export such as ONNX/ncnn/WebGPU only after an export experiment succeeds.
- No FASHN output or training has completed.
- The FASHN still-image endpoint is not a live temporal renderer.

## How to capture the first pilot data

From `C:\Projects\e-commerce`:

```powershell
& .\.venv-vton\Scripts\python.exe .\prototype\local_server.py
```

Open:

```text
http://localhost:8000/prototype/
```

Turn on the camera, select `Reference: neutral clothing`, start recording, move naturally, then stop and save. Repeat with `Wearing: exact suit` while wearing the exact photographed garment. Keep the downloaded files outside Git. Inspect them with the dataset tools before any manifest approval.

## Immediate next tasks

1. Confirm the recorder produces a WebM and JSON sidecar with a nonzero frame count.
2. Import/inspect the two clips with the existing Python/FFmpeg tooling; do not commit personal media.
3. Create restricted rights records and link opaque IDs only after consent and commercial-use scope are documented.
4. Generate the first local FASHN still result and review person preservation, garment detail, orientation, and placement.
5. Validate still-image appearance before attempting temporal/live rendering.

## Verification already run

- `node --check prototype/app.js` passed.
- `python -m py_compile prototype/local_server.py` passed.
- `python -m pytest -q` passed: 14 tests.
