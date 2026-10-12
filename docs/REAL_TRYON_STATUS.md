# Real virtual try-on status

Last updated: 2026-09-29

## Intended feature

The requested product flow is:

1. A shopper selects a garment image.
2. The shopper provides a camera video or photo.
3. A private calibration profile can use the full moving video sequence, not just one frame, to preserve that person’s visual appearance and movement context.
4. A virtual try-on model generates the selected kameez/top on that person.
5. The result is presented as visual appearance guidance, not a promise of size, comfort, tailoring, measurements, or physical fit.

This is different from the existing browser pose overlay. The overlay is still only a rough fallback and is not the final try-on engine.

## Completed so far

- Reviewed online virtual try-on options and selected FASHN VTON v1.5 as the first local research integration because its official repository accepts a person image plus a garment image and supports the `tops` category.
- Downloaded the official FASHN VTON v1.5 source into `vendor/fashn-vton-1.5`.
- Created `.venv-vton` and installed the FASHN package and its Python dependencies.
- CUDA is allowed for private research/training and local evaluation. The local server supports explicit `auto`, `cuda`, and `cpu` modes, while the future API/frontend deployment must not require server CUDA. A portable ONNX/ncnn/WebGPU export has not been validated.
- Validated the user-supplied test video as an 18.715-second portrait MP4 with original audio. A separate external video-edit submission for a kameez-only replacement was rejected before generation because that service requires a Plus plan; no edited video was created.
- Kept the user-supplied product cutout marked as AI-assisted, unverified, private-test-only, and not ground truth.
- Downloaded the FASHN VTON checkpoint and DWPose weights into the ignored local weights directory.
- Added a loopback-only `prototype/local_server.py` endpoint that accepts one camera frame and one garment image in memory and returns one generated still; it does not save or upload images.
- Replaced the inaccurate landmark rectangle overlay with a clearly labeled upright pose-guided garment guide. It is only a temporary visibility fallback and is not a fitted result.
- Added an explicit browser full-video calibration recorder with two stages: `reference_neutral_clothing` and `wearing_exact_garment`.
- The recorder preserves the complete browser WebM until the user clicks **Stop & save full recording**, then downloads the video and a JSON sidecar containing observed presentation timestamps, frame count, dimensions, duration, and SHA-256. Camera-on alone does not save frames.

## Not completed yet

- The FASHN model weights and DWPose weights are downloaded locally. The PyTorch policy block is resolved, but a complete FASHN model initialization and generated try-on image have not yet been verified.
- No real FASHN try-on image has been generated locally yet.
- The local endpoint can now load the PyTorch runtime; no backend/ecommerce integration has been connected, and no generated FASHN output has yet been reviewed.
- The camera page now shows live video with the temporary guide when pose landmarks are available and provides a fitted-still action for the local FASHN path.
- No real calibration video has been confirmed or imported yet; no body/person profile has been trained.
- No claim is made that a live, photorealistic, frame-by-frame try-on is working.

## Next implementation order

1. Capture and save one short reference video and one exact-garment wearing video using the browser recorder; retain the downloaded WebM and JSON sidecars outside Git.
2. Inspect/import those videos with the dataset tools, create QA reports, and link them only after rights records are complete.
3. Run one still-image try-on using a camera frame and the selected kameez image.
4. Review the result for orientation, person preservation, garment detail, and placement; do not accept it as exact physical fit without evidence.
5. Measure CPU latency and resolve a compatible GPU deployment before attempting continuous live video.
6. Review all upstream and third-party licenses before any commercial launch.

## Rights and safety boundary

Public virtual try-on datasets and checkpoints may be research-only or may include third-party fashion images. They must not be placed into an approved commercial training manifest without a rights review. A participant video is a test/reference input unless the required consent and commercial-training permissions are recorded.
