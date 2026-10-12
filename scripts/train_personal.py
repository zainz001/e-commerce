#!/usr/bin/env python3
"""Train a personalized neural try-on model on your video dataset.

Learns the mapping from:
  [Clothing-Agnostic Body (Face + Hands + Background) + Garment Mask + Garment Texture]
to:
  [Photorealistic Render of You Actually Wearing the Cloth]
with natural fabric drape, shadows, folds, and room lighting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ==============================================================================
# Model Architecture: High-Performance U-Net with ResNet Bottleneck
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x)))))


class TryOnGenerator(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 3, base_channels: int = 64, num_res_blocks: int = 4):
        super().__init__()
        # Initial projection
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck Residual Blocks
        res_blocks = [ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)]
        self.bottleneck = nn.Sequential(*res_blocks)

        # Upsampling with skip connections
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final reconstruction layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, agnostic: torch.Tensor, mask: torch.Tensor, garment: torch.Tensor) -> torch.Tensor:
        # Input tensor shape: [B, 7, H, W]
        x_in = torch.cat([agnostic, mask, garment], dim=1)

        d0 = self.in_conv(x_in)       # [B, 64, H, W]
        d1 = self.down1(d0)           # [B, 128, H/2, W/2]
        d2 = self.down2(d1)           # [B, 256, H/4, W/4]

        b = self.bottleneck(d2)       # [B, 256, H/4, W/4]

        u2 = self.up2(b)              # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, d1], dim=1)  # Skip connection: [B, 256, H/2, W/2]

        u1 = self.up1(u2)             # [B, 64, H, W]
        u1 = torch.cat([u1, d0], dim=1)  # Skip connection: [B, 128, H, W]

        out = self.out_conv(u1)       # [B, 3, H, W] in [0, 1]
        return out


# ==============================================================================
# Dataset Loader
# ==============================================================================

class PersonalTryOnDataset(Dataset):
    def __init__(self, data_dir: Path, target_size: tuple[int, int] = (480, 848)):
        self.data_dir = data_dir
        self.target_size = target_size  # (H, W)

        manifest_file = data_dir / "dataset_index.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = data["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        h, w = self.target_size

        frame_p = self.data_dir / item["frame_file"]
        agnostic_p = self.data_dir / item["agnostic_file"]
        mask_p = self.data_dir / item["mask_file"]

        # Load images
        frame = cv2.cvtColor(cv2.imread(str(frame_p)), cv2.COLOR_BGR2RGB)
        agnostic = cv2.cvtColor(cv2.imread(str(agnostic_p)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # Resize to network dimensions (multiple of 16)
        # 480x848 -> (480, 848)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        agnostic = cv2.resize(agnostic, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Garment conditioning: the worn suit region in this frame
        # (For training, the ground-truth garment appearance is within the mask)
        garment = frame.copy()
        garment[mask == 0] = 0

        # Convert to tensors [0, 1]
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        agnostic_t = torch.from_numpy(agnostic).permute(2, 0, 1).float() / 255.0
        garment_t = torch.from_numpy(garment).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return {
            "frame": frame_t,
            "agnostic": agnostic_t,
            "mask": mask_t,
            "garment": garment_t,
        }


# ==============================================================================
# Loss Functions
# ==============================================================================

def image_gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return dx, dy


def compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    # 1. Clothing Region Reconstruction Loss (learns folds, fabric, color)
    cloth_l1 = F.l1_loss(pred * mask, target * mask)

    # 2. Identity & Background Preservation Loss (strictly preserves face, arms, room)
    body_bg_mask = 1.0 - mask
    body_l1 = F.l1_loss(pred * body_bg_mask, target * body_bg_mask) * 2.5

    # 3. Gradient / Edge Loss for sharp fabric folds and embroidery
    p_dx, p_dy = image_gradients(pred * mask)
    t_dx, t_dy = image_gradients(target * mask)
    grad_loss = (F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)) * 0.5

    total_loss = cloth_l1 + body_l1 + grad_loss
    return total_loss, {
        "cloth_l1": cloth_l1.item(),
        "body_l1": body_l1.item(),
        "grad": grad_loss.item(),
        "total": total_loss.item(),
    }


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device_obj}", flush=True)

    dataset = PersonalTryOnDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print(f"Loaded {len(dataset)} samples. Training for {epochs} epochs (batch size: {batch_size})...", flush=True)

    model = TryOnGenerator().to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_obj.type == "cuda"))

    model.train()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            frame = batch["frame"].to(device_obj, non_blocking=True)
            agnostic = batch["agnostic"].to(device_obj, non_blocking=True)
            mask = batch["mask"].to(device_obj, non_blocking=True)
            garment = batch["garment"].to(device_obj, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device_obj.type == "cuda")):
                pred = model(agnostic, mask, garment)
                loss, loss_dict = compute_loss(pred, frame, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(
                total=f"{loss_dict['total']:.3f}",
                cloth=f"{loss_dict['cloth_l1']:.3f}",
                body=f"{loss_dict['body_l1']:.3f}",
            )

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {epoch_loss:.4f}", flush=True)

        # Save checkpoint
        checkpoint_path = output_dir / "personal_tryon_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, checkpoint_path)

    print(f"\nTraining complete! Checkpoint saved to: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train personal neural try-on model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/derived/personal_dataset"),
        help="Path to prepared dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/personal_model"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

