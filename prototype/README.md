# Private camera overlay prototype

This is a private local still-image virtual try-on prototype for the user-supplied Elaf suit cutout. The primary path uses FASHN VTON to synthesize the garment onto a camera frame; it is not a production virtual try-on model, does not estimate size or physical fit, and does not upload or save camera frames.

From the repository root, start the combined static site and local still-image try-on server with the FASHN virtual environment:

```powershell
& .\.venv-vton\Scripts\python.exe .\prototype\local_server.py
```

The private research server defaults to `--device auto` and can use CUDA for local model evaluation. Use `--device cpu` only for a deliberate CPU comparison. This server is not the final deployment runtime; the future API/frontend path must use a validated portable export rather than require server CUDA.

Open `http://localhost:8000/prototype/` in a modern browser, click **Turn on camera**, and allow camera access. Stand facing the camera with your upper body visible, then click **Generate fitted still**. The person frame and garment image are sent only to the loopback server, processed in memory, and returned to the browser. Click **Turn off camera** when finished.

For the first dataset/calibration milestone, choose either **Reference: neutral clothing** or **Wearing: exact suit**, then click **Start full-video recording**. The browser records the complete local video until **Stop & save full recording** is clicked, then downloads the preserved recording plus JSON frame-timing metadata. No recording is uploaded automatically.

The local server requires the FASHN weights under `vendor/fashn-vton-1.5/weights/`. Download them with:

```powershell
& .\.venv-vton\Scripts\python.exe .\vendor\fashn-vton-1.5\scripts\download_weights.py --weights-dir .\vendor\fashn-vton-1.5\weights
```

The page must be served from localhost because browsers normally restrict camera access from a `file://` page. Camera permission is requested only after the button is clicked and the stream stays in the browser tab. The upright pose guide is only a temporary visibility fallback; it is not the final fitted garment renderer.

The referenced cutout is AI-assisted, source-rights-unverified, and marked private-test-only. Do not put it or any camera capture into an approved training or commercial manifest. The fitted still is a research prototype and must not be described as proof of exact size, comfort, tailoring, measurements, stretch, or physical fit. Continuous fitted video is not implemented yet.
