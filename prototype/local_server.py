#!/usr/bin/env python3
"""Serve the private prototype and run local still-image virtual try-on.

The server binds to loopback only. Images are accepted in memory and are not
written to disk or sent to a hosted inference service.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import sys
from threading import Lock
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "vendor" / "fashn-vton-1.5" / "weights"
MAX_REQUEST_BYTES = 24 * 1024 * 1024
DEFAULT_DEVICE = "auto"


def decode_data_url(value: str) -> bytes:
    """Decode a base64 data URL without accepting arbitrary URLs."""

    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("image must be a base64 data URL")
    try:
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("image data URL must use base64")
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid image data URL") from error


class TryOnHandler(SimpleHTTPRequestHandler):
    """Static-file handler with one local try-on endpoint."""

    pipeline: Any = None
    pipeline_lock = Lock()
    device = DEFAULT_DEVICE
    weights_dir: Path = DEFAULT_WEIGHTS

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory or str(ROOT), **kwargs)

    def end_headers(self) -> None:
        if self.path.startswith("/prototype/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8000")
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _get_pipeline(cls) -> Any:
        if cls.pipeline is not None:
            return cls.pipeline
        with cls.pipeline_lock:
            if cls.pipeline is not None:
                return cls.pipeline
            if not (cls.weights_dir / "model.safetensors").is_file():
                raise RuntimeError(
                    f"FASHN weights are missing at {cls.weights_dir}. "
                    "Run vendor/fashn-vton-1.5/scripts/download_weights.py first."
                )
            try:
                import torch
                from fashn_vton import TryOnPipeline
            except Exception as error:  # pragma: no cover - depends on the local venv
                raise RuntimeError(
                    "FASHN cannot be imported. Start this server with .venv-vton\\Scripts\\python.exe."
                ) from error
            if cls.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was explicitly requested but is unavailable")
                device = "cuda"
            elif cls.device == "cpu":
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading FASHN on {device} (device mode: {cls.device}).", flush=True)
            cls.pipeline = TryOnPipeline(weights_dir=str(cls.weights_dir), device=device)
        return cls.pipeline

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8000")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    neural_model: Any = None
    cached_garment_src: str | None = None
    cached_garment_rgba: Any = None

    def _handle_live_tryon(self, request: dict[str, Any]) -> None:
        import cv2
        import numpy as np
        from prototype.tryon_engine import parse_landmarks, skeleton_warp_garment, realistic_tryon_blend

        person_bytes = decode_data_url(request["person_image"])
        person_nparr = np.frombuffer(person_bytes, np.uint8)
        person_bgr = cv2.imdecode(person_nparr, cv2.IMREAD_COLOR)
        if person_bgr is None:
            raise ValueError("Could not decode person camera frame")
        h, w = person_bgr.shape[:2]

        garment_src = request.get("garment_image")
        if garment_src and garment_src == TryOnHandler.cached_garment_src and TryOnHandler.cached_garment_rgba is not None:
            garment_rgba = TryOnHandler.cached_garment_rgba
        elif garment_src:
            garment_bytes = decode_data_url(garment_src)
            garment_nparr = np.frombuffer(garment_bytes, np.uint8)
            garment_raw = cv2.imdecode(garment_nparr, cv2.IMREAD_UNCHANGED)
            if garment_raw is not None:
                if len(garment_raw.shape) == 3 and garment_raw.shape[2] == 4:
                    garment_rgba = garment_raw
                else:
                    # If 3 channels (no alpha), auto-detect white/light background
                    bgr = garment_raw[:, :, :3]
                    is_bg = (bgr[:, :, 0] > 235) & (bgr[:, :, 1] > 235) & (bgr[:, :, 2] > 235)
                    alpha = (~is_bg).astype(np.uint8) * 255
                    garment_rgba = np.dstack([bgr, alpha])
                TryOnHandler.cached_garment_src = garment_src
                TryOnHandler.cached_garment_rgba = garment_rgba
            else:
                garment_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            garment_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        mask_src = request.get("body_mask")
        if mask_src:
            mask_bytes = decode_data_url(mask_src)
            mask_nparr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        if np.count_nonzero(mask) < 100:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(h * 0.18):int(h * 0.88), int(w * 0.2):int(w * 0.8)] = 255

        # Extract landmarks and warp garment to human skeleton
        landmarks = request.get("landmarks")
        kp = parse_landmarks(landmarks, h, w, mask)
        warped = skeleton_warp_garment(garment_rgba, kp, h, w)

        # Apply physical fold lighting and seamless soft-edge composite
        out_bgr = realistic_tryon_blend(person_bgr, warped, mask)

        _, buf = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded = base64.b64encode(buf).decode("ascii")
        self._write_json({"image": f"data:image/jpeg;base64,{encoded}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path not in ("/api/tryon", "/api/live_neural_tryon"):
            self._write_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._write_json({"error": "request_too_large_or_empty"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        try:
            request = json.loads(self.rfile.read(content_length))
            if self.path == "/api/live_neural_tryon":
                self._handle_live_tryon(request)
                return
            from PIL import Image, ImageOps

            person = ImageOps.exif_transpose(Image.open(io.BytesIO(decode_data_url(request["person_image"]))))
            garment = ImageOps.exif_transpose(Image.open(io.BytesIO(decode_data_url(request["garment_image"]))))
            person = person.convert("RGB")
            garment = garment.convert("RGB")

            result = self._get_pipeline()(
                person_image=person,
                garment_image=garment,
                category="tops",
                garment_photo_type="flat-lay",
                num_samples=1,
                num_timesteps=30,
                guidance_scale=1.5,
                seed=42,
                segmentation_free=True,
            )
            output = io.BytesIO()
            result.images[0].save(output, format="PNG")
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            self._write_json({"image": f"data:image/png;base64,{encoded}"})
        except KeyError as error:
            self._write_json({"error": f"missing field: {error.args[0]}"}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - model/runtime errors are environment-specific
            self.log_error("Local try-on failed: %s", error)
            self._write_json({"error": str(error)[:500]}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Private local FASHN try-on server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; loopback is the safe default")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=DEFAULT_DEVICE,
        help="Research inference device; final exported runtimes must not depend on server CUDA",
    )
    args = parser.parse_args()

    TryOnHandler.weights_dir = args.weights_dir.resolve()
    TryOnHandler.device = args.device
    server = ThreadingHTTPServer((args.host, args.port), TryOnHandler)
    print(f"Serving the private prototype at http://{args.host}:{args.port}/prototype/")
    print(f"FASHN weights: {TryOnHandler.weights_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopping local prototype server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
