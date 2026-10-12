#!/usr/bin/env python3
"""Extract frames and generate clothing-agnostic representations from personal video.

Reads the user's video wearing kameez shalwar, extracts frames, and uses
FashnHumanParser to segment the garment region from person identity and background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def prepare_dataset(
    video_path: Path,
    output_dir: Path,
    sample_stride: int = 2,
    max_frames: int = 300,
    device: str = "cuda",
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found at: {video_path}")

    frames_dir = output_dir / "frames"
    agnostic_dir = output_dir / "agnostic"
    masks_dir = output_dir / "masks"
    garment_crops_dir = output_dir / "garment_crops"

    for d in (frames_dir, agnostic_dir, masks_dir, garment_crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading FashnHumanParser on {device}...", flush=True)
    import torch
    from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", flush=True)
        device = "cpu"

    parser = FashnHumanParser(device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video opened: {total_frames} frames @ {fps:.1f} fps", flush=True)

    cloth_labels = {
        LABELS_TO_IDS[k]
        for k in ("top", "dress", "pants", "skirt", "scarf", "torso")
        if k in LABELS_TO_IDS
    }

    metadata = []
    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=min(total_frames // sample_stride, max_frames), desc="Extracting & Parsing")

    while cap.isOpened() and saved_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            w, h = pil_img.size

            # Run human parser
            seg_map = parser.predict(pil_img)

            # Binary mask for clothing
            cloth_mask = np.isin(seg_map, list(cloth_labels)).astype(np.uint8) * 255

            # Dilate mask slightly to cleanly remove clothing edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dilated_cloth_mask = cv2.dilate(cloth_mask, kernel, iterations=1)

            # Create agnostic image: person face, hands, hair & background intact, cloth infilled with neutral gray
            agnostic = frame_rgb.copy()
            agnostic[dilated_cloth_mask > 0] = [128, 128, 128]

            # Crop garment region if present
            ys, xs = np.where(cloth_mask > 0)
            if len(xs) > 100:
                x_min, x_max = max(0, xs.min() - 5), min(w, xs.max() + 5)
                y_min, y_max = max(0, ys.min() - 5), min(h, ys.max() + 5)
                garment_crop = frame_rgb[y_min:y_max, x_min:x_max]
            else:
                garment_crop = frame_rgb

            stem = f"frame_{saved_count:04d}"
            frame_path = frames_dir / f"{stem}.jpg"
            agnostic_path = agnostic_dir / f"{stem}.jpg"
            mask_path = masks_dir / f"{stem}.png"
            crop_path = garment_crops_dir / f"{stem}.jpg"

            Image.fromarray(frame_rgb).save(frame_path, quality=95)
            Image.fromarray(agnostic).save(agnostic_path, quality=95)
            Image.fromarray(dilated_cloth_mask).save(mask_path)
            Image.fromarray(garment_crop).save(crop_path, quality=90)

            metadata.append({
                "index": saved_count,
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 4),
                "frame_file": str(frame_path.relative_to(output_dir)),
                "agnostic_file": str(agnostic_path.relative_to(output_dir)),
                "mask_file": str(mask_path.relative_to(output_dir)),
                "crop_file": str(crop_path.relative_to(output_dir)),
                "width": w,
                "height": h,
            })

            saved_count += 1
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    manifest_file = output_dir / "dataset_index.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({"total_samples": saved_count, "samples": metadata}, f, indent=2)

    print(f"\nDone! Processed {saved_count} frames.")
    print(f"Dataset saved to: {output_dir}")
    print(f"Manifest written to: {manifest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and parse personal video dataset")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("WhatsApp Video 2026-09-27 at 6.59.51 PM.mp4"),
        help="Path to personal video",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/personal_dataset"),
        help="Directory to save extracted dataset",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Extract every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device for parser inference",
    )
    args = parser.parse_args()

    prepare_dataset(
        video_path=args.video,
        output_dir=args.output_dir,
        sample_stride=args.stride,
        max_frames=args.max_frames,
        device=args.device,
    )


if __name__ == "__main__":
    main()
