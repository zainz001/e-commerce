# Changelog

## 2026-09-29

### Changed

- Corrected the runtime boundary after clarification: CUDA is allowed for private training/research and local evaluation, while the future API/frontend runtime must not depend on server CUDA. The local server now accepts explicit `--device auto|cuda|cpu` modes; portable export remains unvalidated.
- Confirmed the two local calibration recordings in `C:\Users\uhani\Downloads`: neutral reference (`14.528s`, `217` observed timing entries) and exact-garment wearing (`14.821s`, `222` observed timing entries). Their JSON SHA-256 values match the WebM files; the recordings remain outside the repository.
- Removed the visible prototype cache-busting query string so the stable URL remains `http://localhost:8000/prototype/`, and made the local FASHN server select CUDA automatically when PyTorch reports it available.
- Resolved the local PyTorch runtime block with a narrow Windows Application Control allow rule. The CUDA-enabled wheel is restored for private training/research, while the final API/frontend deployment remains required to use a validated CUDA-independent export.
- Recorded owner approval for the visually convincing exact-garment try-on objective and initial pilot boundaries.
- Clarified that “fitting” means visual wearing appearance only; physical sizing and fit claims remain out of scope.
- Recorded the remaining prerequisite to specify restricted consent-record ownership, storage, retention, withdrawal, and access controls before approving real media.
- Downloaded the local FASHN VTON and DWPose weights; human-parser initialization remains blocked by the Windows Application Control policy rejecting the installed PyTorch DLL.
- Added a loopback-only FASHN still-image endpoint and camera snapshot control under `prototype/`.
- Removed the misleading landmark rectangle overlay from the primary camera path so the prototype no longer presents visibly upside-down or off-body placement as try-on output.
- Made FASHN initialization single-flight and guarded duplicate browser requests so concurrent camera tabs or an automatic/manual request do not download and load the human-parser/model weights twice. Later requests reuse the in-memory pipeline; Hugging Face files remain in the local cache for later server runs.
- Added cache-busting and no-store behavior for prototype assets, broadened the first camera request for desktop compatibility, and added explicit permission/security/busy-camera error messages.
- Made the first fitted-still request start automatically after camera permission and added a clear error when the page is served by the static server without the `/api/tryon` endpoint.
- Restored a clearly labeled upright pose-guided garment guide as a temporary fallback so the camera view does not become garment-free while FASHN is unavailable.
- Added an explicit full-video calibration recorder with reference/wearing stages, local WebM capture, observed frame timestamps, SHA-256 metadata, and no-save-on-camera-start behavior.
- Added `docs/NEXT_CHAT_HANDOFF.md` and aligned `AGENTS.md`, `docs/DATA_DECISIONS.md`, `docs/REAL_TRYON_STATUS.md`, `docs/LOCAL_SETUP.md`, and the prototype README with the full-video calibration workflow.

### Verification

- Calibration sidecar/file checksum verification passed for both local WebM recordings.
- Stable prototype asset URLs no longer require a changing query string.
- `python -m py_compile prototype/local_server.py` passed after automatic CUDA-device selection was added.
- `torch 2.14.0+cu130` imports successfully; CUDA 13.0 is available on the detected NVIDIA GeForce RTX 4050 Laptop GPU; `onnxruntime-gpu 1.30.0` exposes CUDA and CPU providers; `14` tests pass. Local server device selection is explicit and the final export path remains unvalidated.
- `node --check prototype/app.js` passed.
- `python -m py_compile prototype/local_server.py` passed.
- FASHN model and DWPose weight files downloaded successfully.
- Local `import torch` verification failed with `WinError 4551` before inference; no try-on image was generated.
- After the recorder changes, `node --check prototype/app.js` passed and `python -m pytest -q` passed with 14 tests.

## 2026-09-27

### Added

- Initial dataset README and scope statement.
- Proposed schema and validation rules in `docs/DATASET_SPEC.md`.
- Capture protocol, consent/rights policy, and annotation guide.
- Initial data decisions and repository protection rules.
- Clearly fictional JSONL manifest example.
- Python package metadata and a read-only `dataset inspect` CLI.
- Recursive SHA-256 inspection, duplicate checksum reporting, and selected `ffprobe` metadata extraction.
- Synthetic tests covering hashing, duplicates, read-only behavior, and invalid input handling.
- Rights-aware JSONL manifest validation with ID, schema, linkage, path-safety, checksum, and approval checks.
- Verified local FFmpeg 9.0.2 Essentials installation under ignored `tools/ffmpeg/`.
- Non-destructive timestamped preview and contact-sheet generation with checksum-linked metadata.
- Privacy-conscious coverage/QA reporting and deterministic grouped split generation.
- Owner approval checklist for scope, consent, commercial rights, storage, retention, and withdrawal decisions.
- Restricted rights-registry cross-checking for active status and commercial-training permission.
- Clarified the product intent as visual try-on of the exact suit on the same person, including natural drape and movement without physical-fit claims.
- Created a transparent AI-assisted garment cutout from the user-supplied product image for private prototype testing; explicitly marked it as unverified and not ground truth.
- Added `prototype/`, a local pose-guided camera approximation that automatically follows shoulder width, torso angle, and movement. It requests camera access only after a user click, keeps the stream in-browser, and does not upload or save frames.
- Corrected the browser module URL for the pose tracker and added a detailed startup error in the private prototype.
- Vendored the official MediaPipe Tasks Vision runtime and Lite pose model under `prototype/vendor` after the external CDN was unavailable in the local environment.
- Started the real virtual try-on integration: downloaded the official FASHN VTON v1.5 source into `vendor/fashn-vton-1.5`, created `.venv-vton`, and installed its package dependencies. The CPU PyTorch runtime is working; the attempted CUDA wheel did not load and GPU support remains unresolved.
- Added `docs/REAL_TRYON_STATUS.md` to distinguish the completed setup from the unfinished model weights, backend, and live-camera work. No real try-on output has been generated yet.
- Probed a user-supplied 18.715-second portrait test video and recorded its kameez-only edit attempt. The external video-edit service rejected generation because the account requires Plus; no output was created.
- Corrected `.gitignore` so local virtual environments, external model source trees, browser runtimes, raw/derived media, and downloaded weights stay out of GitHub while source code, tests, documentation, examples, and project configuration remain committable.

### Verification

- Repository inspection completed; no real dataset media or tooling existed before this change.
- `python -m compileall -q src tests` passed.
- Synthetic inspector smoke check passed.
- CLI smoke check on the test directory passed.
- Synthetic validator checks passed for valid candidates, approval failures, path traversal, and fictional records.
- CLI validation of the fictional example passed with the expected documentation warning.
- FFmpeg archive SHA-256 matched the publisher checksum.
- A synthetic 320×240 H.264 video passed the real `ffprobe` inspector with codec and dimensions recorded.
- `python -m pytest -q` passed: 14 tests.
- A synthetic preview run produced three timestamped JPEG frames and a contact sheet; the second run skipped identical output.
- Windows UTF-8 BOM handling was tested for JSONL manifests.
- Coverage report and grouped split synthetic checks passed.
- Approved-record validation against a synthetic restricted rights registry passed.
- Final CLI sweep passed for version, inspect, validate, report, and automatic local `ffprobe` discovery.
- The package built successfully in editable mode; installation into the managed user site was denied by environment permissions, so the documented checkout workflow uses `PYTHONPATH=src`.

### Unresolved

- The rights-record format and legal review process still need owner approval.
- No real capture has been performed, so coverage and quality are unvalidated.

### Next smallest task

Record owner approval decisions, then run the tools on a small properly consented pilot without committing raw media.
