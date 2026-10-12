#!/usr/bin/env python3
"""Render a personalized video virtual try-on using the trained neural model.

Takes any garment image (e.g. Elaf suit or uploaded suit) and synthesizes it
onto your body across all frames, producing an output MP4 video where you are
actually wearing the suit with realistic fabric drape, folds, and lighting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from train_personal import TryOnGenerator


def load_model(checkpoint_path: Path, device: torch.device) -> TryOnGenerator:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint_path}")

    model = TryOnGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded trained model from: {checkpoint_path}", flush=True)
    return model


def prepare_garment_input(garment_bgr: np.ndarray, mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Warp and align target garment to fill the body's clothing mask bounding box."""
    h, w = target_shape
    garment_rgb = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2RGB)

    # Find clothing region bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return cv2.resize(garment_rgb, (w, h))

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    # Resize garment to fit the body's torso & clothing box
    resized_garment = cv2.resize(garment_rgb, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

    # Create full canvas and place resized garment into position
    aligned = np.zeros((h, w, 3), dtype=np.uint8)
    aligned[y_min : y_min + box_h, x_min : x_min + box_w] = resized_garment
    return aligned


def render_tryon_video(
    data_dir: Path,
    garment_path: Path,
    checkpoint_path: Path,
    output_video_path: Path,
    fps: float = 25.0,
    device: str = "cuda",
) -> None:
    device_obj = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device_obj)

    manifest_file = data_dir / "dataset_index.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    # Load target garment image
    if not garment_path.is_file():
        raise FileNotFoundError(f"Garment image not found: {garment_path}")
    garment_raw = cv2.imread(str(garment_path), cv2.IMREAD_UNCHANGED)
    if garment_raw is None:
        raise ValueError(f"Could not read garment: {garment_path}")

    # If garment has alpha channel, blend onto white/neutral before processing
    if garment_raw.shape[2] == 4:
        alpha = garment_raw[:, :, 3] / 255.0
        garment_bgr = (garment_raw[:, :, :3] * alpha[:, :, None] + (1.0 - alpha[:, :, None]) * 255).astype(np.uint8)
    else:
        garment_bgr = garment_raw

    first_frame_path = data_dir / samples[0]["frame_file"]
    sample_frame = cv2.imread(str(first_frame_path))
    h, w = sample_frame.shape[:2]

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    print(f"Rendering try-on video for {len(samples)} frames -> {output_video_path}...", flush=True)

    with torch.no_grad():
        for item in tqdm(samples, desc="Synthesizing Frames"):
            agnostic_p = data_dir / item["agnostic_file"]
            mask_p = data_dir / item["mask_file"]

            agnostic_bgr = cv2.imread(str(agnostic_p))
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

            # Align target suit into your body's clothing region
            aligned_garment_rgb = prepare_garment_input(garment_bgr, mask, (h, w))
            agnostic_rgb = cv2.cvtColor(agnostic_bgr, cv2.COLOR_BGR2RGB)

            # Convert to PyTorch tensors
            agnostic_t = torch.from_numpy(agnostic_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device_obj) / 255.0
            garment_t = torch.from_numpy(aligned_garment_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device_obj) / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device_obj) / 255.0

            # Generate photorealistic frame with neural model
            with torch.cuda.amp.autocast(enabled=(device_obj.type == "cuda")):
                pred_t = model(agnostic_t, mask_t, garment_t)

            pred_np = (pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

            out.write(pred_bgr)

    out.release()
    print(f"\nDone! High-quality try-on video saved to: {output_video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render personal virtual try-on video")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/derived/personal_dataset"),
        help="Path to prepared dataset directory",
    )
    parser.add_argument(
        "--garment",
        type=Path,
        default=Path("data/derived/product_cutouts/elaf_premium_10_front_cutout.png"),
        help="Path to garment design image",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/personal_model/personal_tryon_model.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/output_tryon_video.mp4"),
        help="Path to output video file",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    render_tryon_video(
        data_dir=args.data_dir,
        garment_path=args.garment,
        checkpoint_path=args.checkpoint,
        output_video_path=args.output,
        fps=args.fps,
        device=args.device,
    )


if __name__ == "__main__":
    main()

