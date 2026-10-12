# Local setup

This project uses local FFmpeg/`ffprobe` for source inspection and derived previews. Media and processing stay on the local machine; no source footage is uploaded by the dataset tools.

## Verified Windows installation

The current workspace has a private, Git-ignored installation at:

```text
tools/ffmpeg/bin/ffmpeg.exe
tools/ffmpeg/bin/ffprobe.exe
```

Verified on 2026-09-27:

- FFmpeg: `9.0.2-essentials_build-www.gyan.dev`
- Archive: `ffmpeg-9.0.2-essentials_build.7z`
- SHA-256: `4705843ccaaf54257c16ad90f3e952ece33c17df964ecf7bfdbb0f49c7171077`
- Publisher/build source: [FFmpeg Windows builds from gyan.dev](https://www.gyan.dev/ffmpeg/builds/), which is linked by the official [FFmpeg download page](https://ffmpeg.org/download.html).

The archive and extraction temporary files were removed after verification. The installed binaries remain outside Git under `tools/ffmpeg/`.

## Using the local binaries

The CLI searches `PATH` by default. Until this private folder is added to the current shell's `PATH`, pass the executable explicitly:

```powershell
$env:PYTHONPATH = 'src'
python -m dataset inspect --input <source-directory> --ffprobe .\tools\ffmpeg\bin\ffprobe.exe --dry-run --output inspection.json
```

The portable checkout workflow uses `PYTHONPATH=src`. An editable install may be used in a normal Python environment, but the managed Python environment used for this repository denied creation of its user-site `.pth` file.

Do not add `tools/ffmpeg/` to a commit. Reinstall or replace it only after verifying the new publisher checksum and updating this record.

## Private prototype and calibration capture

From the repository root, start the loopback-only prototype server:

```powershell
& .\.venv-vton\Scripts\python.exe .\prototype\local_server.py
```

Open `http://localhost:8000/prototype/`. Camera startup only displays the live stream. To capture the full temporal sequence, choose a calibration stage, click **Start full-video recording**, move naturally, and click **Stop & save full recording**. The browser downloads a `.webm` recording and a `.json` sidecar with observed frame timestamps and a SHA-256 checksum. No recording is uploaded automatically. Keep those files outside Git and do not add them to an approved manifest until rights and QA validation are complete.

The FASHN weights are expected under `vendor/fashn-vton-1.5/weights/`. The private research server defaults to automatic device selection and can use CUDA for local evaluation. Start it with `--device cuda` to require CUDA or `--device cpu` for a CPU comparison. The pipeline is loaded once into process memory and reused for later requests. The future API/frontend deployment must not depend on server CUDA; portable export remains unvalidated.
