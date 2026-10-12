#!/usr/bin/env python3
"""High-performance real-time virtual try-on engine for live camera video.

Deforms garment textures to body skeleton landmarks (shoulders, arms, torso, legs),
infills the wearer's exact silhouette, applies physical fabric lighting and fold shading,
and composites seamlessly onto the live video feed.
"""

from __future__ import annotations

import cv2
import numpy as np


def parse_landmarks(
    landmarks_input: list[dict[str, float]] | list[list[float]] | np.ndarray | None,
    h: int,
    w: int,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Parse MediaPipe (33) or DWPose (18) landmarks into standardized pixel coordinates."""
    kp: dict[str, np.ndarray] = {}

    if landmarks_input is not None and len(landmarks_input) >= 18:
        # Check if MediaPipe 33 format
        if len(landmarks_input) >= 33 or (isinstance(landmarks_input[0], dict) and "visibility" in landmarks_input[0]):
            # MediaPipe format: dicts with x, y
            def get_mp(idx: int) -> np.ndarray:
                item = landmarks_input[idx]
                if isinstance(item, dict):
                    return np.array([item["x"] * w, item["y"] * h], dtype=np.float32)
                return np.array([item[0] * w, item[1] * h], dtype=np.float32)

            # In natural camera frame:
            # MediaPipe 12: right shoulder, 11: left shoulder
            # 14: right elbow, 13: left elbow
            # 16: right wrist, 15: left wrist
            # 24: right hip, 23: left hip
            # 26: right knee, 25: left knee
            # 28: right ankle, 27: left ankle
            kp["r_sh"] = get_mp(12)
            kp["l_sh"] = get_mp(11)
            kp["neck"] = (kp["r_sh"] + kp["l_sh"]) * 0.5
            kp["r_elb"] = get_mp(14)
            kp["l_elb"] = get_mp(13)
            kp["r_wri"] = get_mp(16)
            kp["l_wri"] = get_mp(15)
            kp["r_hip"] = get_mp(24)
            kp["l_hip"] = get_mp(23)
            kp["r_knee"] = get_mp(26)
            kp["l_knee"] = get_mp(25)
            kp["r_ank"] = get_mp(28)
            kp["l_ank"] = get_mp(27)
        else:
            # DWPose / COCO format: 1: Neck, 2: RSh, 3: RElb, 4: RWri, 5: LSh, 6: LElb, 7: LWri, 8: RHip, 9: RKnee, 10: RAnk, 11: LHip, 12: LKnee, 13: LAnk
            def get_dw(idx: int) -> np.ndarray:
                item = landmarks_input[idx]
                if isinstance(item, dict):
                    return np.array([item["x"] * w, item["y"] * h], dtype=np.float32)
                return np.array([item[0] * w, item[1] * h], dtype=np.float32)

            kp["neck"] = get_dw(1)
            kp["r_sh"] = get_dw(2)
            kp["r_elb"] = get_dw(3)
            kp["r_wri"] = get_dw(4)
            kp["l_sh"] = get_dw(5)
            kp["l_elb"] = get_dw(6)
            kp["l_wri"] = get_dw(7)
            kp["r_hip"] = get_dw(8)
            kp["r_knee"] = get_dw(9)
            kp["r_ank"] = get_dw(10)
            kp["l_hip"] = get_dw(11)
            kp["l_knee"] = get_dw(12)
            kp["l_ank"] = get_dw(13)

    # Fallback if landmarks missing or invalid: estimate from body mask
    if "r_sh" not in kp or np.isnan(kp["r_sh"][0]):
        if mask is not None and np.count_nonzero(mask > 80) > 100:
            ys, xs = np.where(mask > 80)
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            bw, bh = x1 - x0, y1 - y0
            mid_x = (x0 + x1) * 0.5
            kp["neck"] = np.array([mid_x, y0 + bh * 0.12], dtype=np.float32)
            kp["r_sh"] = np.array([mid_x - bw * 0.35, y0 + bh * 0.16], dtype=np.float32)
            kp["l_sh"] = np.array([mid_x + bw * 0.35, y0 + bh * 0.16], dtype=np.float32)
            kp["r_elb"] = np.array([mid_x - bw * 0.45, y0 + bh * 0.38], dtype=np.float32)
            kp["l_elb"] = np.array([mid_x + bw * 0.45, y0 + bh * 0.38], dtype=np.float32)
            kp["r_wri"] = np.array([mid_x - bw * 0.40, y0 + bh * 0.58], dtype=np.float32)
            kp["l_wri"] = np.array([mid_x + bw * 0.40, y0 + bh * 0.58], dtype=np.float32)
            kp["r_hip"] = np.array([mid_x - bw * 0.25, y0 + bh * 0.55], dtype=np.float32)
            kp["l_hip"] = np.array([mid_x + bw * 0.25, y0 + bh * 0.55], dtype=np.float32)
            kp["r_knee"] = np.array([mid_x - bw * 0.22, y0 + bh * 0.85], dtype=np.float32)
            kp["l_knee"] = np.array([mid_x + bw * 0.22, y0 + bh * 0.85], dtype=np.float32)
            kp["r_ank"] = np.array([mid_x - bw * 0.20, y1], dtype=np.float32)
            kp["l_ank"] = np.array([mid_x + bw * 0.20, y1], dtype=np.float32)
        else:
            kp["neck"] = np.array([w * 0.50, h * 0.22], dtype=np.float32)
            kp["r_sh"] = np.array([w * 0.30, h * 0.26], dtype=np.float32)
            kp["l_sh"] = np.array([w * 0.70, h * 0.26], dtype=np.float32)
            kp["r_elb"] = np.array([w * 0.20, h * 0.44], dtype=np.float32)
            kp["l_elb"] = np.array([w * 0.80, h * 0.44], dtype=np.float32)
            kp["r_wri"] = np.array([w * 0.22, h * 0.62], dtype=np.float32)
            kp["l_wri"] = np.array([w * 0.78, h * 0.62], dtype=np.float32)
            kp["r_hip"] = np.array([w * 0.35, h * 0.60], dtype=np.float32)
            kp["l_hip"] = np.array([w * 0.65, h * 0.60], dtype=np.float32)
            kp["r_knee"] = np.array([w * 0.35, h * 0.85], dtype=np.float32)
            kp["l_knee"] = np.array([w * 0.65, h * 0.85], dtype=np.float32)
            kp["r_ank"] = np.array([w * 0.35, h * 0.98], dtype=np.float32)
            kp["l_ank"] = np.array([w * 0.65, h * 0.98], dtype=np.float32)

    return kp


def skeleton_warp_garment(
    garment_rgba: np.ndarray,
    kp: dict[str, np.ndarray],
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Warp garment semantic regions (torso, sleeves, pants) onto wearer skeleton."""
    gh, gw = garment_rgba.shape[:2]

    neck = kp["neck"]
    r_sh, l_sh = kp["r_sh"], kp["l_sh"]
    r_elb, l_elb = kp["r_elb"], kp["l_elb"]
    r_wri, l_wri = kp["r_wri"], kp["l_wri"]
    r_knee, l_knee = kp.get("r_knee", r_sh + np.array([0, 300])), kp.get("l_knee", l_sh + np.array([0, 300]))
    r_ank, l_ank = kp.get("r_ank", r_knee + np.array([0, 150])), kp.get("l_ank", l_knee + np.array([0, 150]))

    sh_vec = l_sh - r_sh
    sh_width = float(np.linalg.norm(sh_vec)) or 100.0
    sh_dir = sh_vec / sh_width
    perp = np.array([-sh_dir[1], sh_dir[0]])
    if perp[1] < 0:
        perp = -perp

    # 1. Torso Quad
    src_torso = np.float32([
        [gw * 0.28, gh * 0.02],
        [gw * 0.72, gh * 0.02],
        [gw * 0.75, gh * 0.78],
        [gw * 0.25, gh * 0.78],
    ])
    dst_torso = np.float32([
        r_sh - perp * (sh_width * 0.22) - sh_dir * (sh_width * 0.15),
        l_sh - perp * (sh_width * 0.22) + sh_dir * (sh_width * 0.15),
        l_knee + sh_dir * (sh_width * 0.35),
        r_knee - sh_dir * (sh_width * 0.35),
    ])
    M_torso = cv2.getPerspectiveTransform(src_torso, dst_torso)
    warped_torso = cv2.warpPerspective(garment_rgba, M_torso, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # 2. Right Sleeve (viewer's left = person's right)
    src_r_sleeve = np.float32([
        [gw * 0.33, gh * 0.04],
        [gw * 0.35, gh * 0.36],
        [gw * 0.24, gh * 0.78],
        [gw * 0.03, gh * 0.65],
    ])
    r_arm_vec = r_wri - r_sh
    r_arm_len = float(np.linalg.norm(r_arm_vec)) or 100.0
    r_arm_dir = r_arm_vec / r_arm_len
    r_arm_perp = np.array([-r_arm_dir[1], r_arm_dir[0]])
    if np.dot(r_arm_perp, -sh_dir) < 0:
        r_arm_perp = -r_arm_perp

    cuff_w = sh_width * 0.60
    dst_r_sleeve = np.float32([
        r_sh - sh_dir * (sh_width * 0.25) - perp * (sh_width * 0.20),
        r_sh + perp * (sh_width * 0.55),
        r_wri - r_arm_perp * (cuff_w * 0.5),
        r_wri + r_arm_perp * (cuff_w * 1.2),
    ])
    M_r_sleeve = cv2.getPerspectiveTransform(src_r_sleeve, dst_r_sleeve)
    warped_r_sleeve = cv2.warpPerspective(garment_rgba, M_r_sleeve, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # 3. Left Sleeve (viewer's right = person's left)
    src_l_sleeve = np.float32([
        [gw * 0.67, gh * 0.04],
        [gw * 0.65, gh * 0.36],
        [gw * 0.76, gh * 0.78],
        [gw * 0.97, gh * 0.65],
    ])
    l_arm_vec = l_wri - l_sh
    l_arm_len = float(np.linalg.norm(l_arm_vec)) or 100.0
    l_arm_dir = l_arm_vec / l_arm_len
    l_arm_perp = np.array([-l_arm_dir[1], l_arm_dir[0]])
    if np.dot(l_arm_perp, sh_dir) < 0:
        l_arm_perp = -l_arm_perp

    dst_l_sleeve = np.float32([
        l_sh + sh_dir * (sh_width * 0.25) - perp * (sh_width * 0.20),
        l_sh + perp * (sh_width * 0.55),
        l_wri - l_arm_perp * (cuff_w * 0.5),
        l_wri + l_arm_perp * (cuff_w * 1.2),
    ])
    M_l_sleeve = cv2.getPerspectiveTransform(src_l_sleeve, dst_l_sleeve)
    warped_l_sleeve = cv2.warpPerspective(garment_rgba, M_l_sleeve, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # 4. Pants / Shalwar
    src_pants = np.float32([
        [gw * 0.30, gh * 0.78],
        [gw * 0.70, gh * 0.78],
        [gw * 0.70, gh * 0.99],
        [gw * 0.30, gh * 0.99],
    ])
    dst_pants = np.float32([
        r_knee - sh_dir * (sh_width * 0.2),
        l_knee + sh_dir * (sh_width * 0.2),
        l_ank + sh_dir * (sh_width * 0.2),
        r_ank - sh_dir * (sh_width * 0.2),
    ])
    M_pants = cv2.getPerspectiveTransform(src_pants, dst_pants)
    warped_pants = cv2.warpPerspective(garment_rgba, M_pants, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # Fast boolean composite (pants < sleeves < torso)
    combined = warped_pants.copy()
    m_r = warped_r_sleeve[:, :, 3] > 40
    combined[m_r] = warped_r_sleeve[m_r]
    m_l = warped_l_sleeve[:, :, 3] > 40
    combined[m_l] = warped_l_sleeve[m_l]
    m_t = warped_torso[:, :, 3] > 40
    combined[m_t] = warped_torso[m_t]

    return combined


def realistic_tryon_blend(
    frame_bgr: np.ndarray,
    warped_rgba: np.ndarray,
    cloth_mask: np.ndarray,
) -> np.ndarray:
    """Seamless physical blending with lighting, fold shading, and modesty infilling."""
    h, w = frame_bgr.shape[:2]
    warped_rgb = warped_rgba[:, :, :3].copy()
    garment_alpha = warped_rgba[:, :, 3]

    # Infill: Ensure 100% of the clothing silhouette is populated with garment fabric
    mask_bool = (cloth_mask > 80)
    uncovered = mask_bool & (garment_alpha < 60)
    if np.any(uncovered):
        covered = (garment_alpha >= 100)
        if np.any(covered):
            fabric_color = np.median(warped_rgb[covered], axis=0).astype(np.uint8)
            warped_rgb[uncovered] = fabric_color

    # 1. Physical Lighting & Fold Shading Extraction
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cloth_pix = gray[mask_bool]
    if len(cloth_pix) > 0:
        mean_l = float(np.median(cloth_pix))
        shading = np.clip(gray / max(mean_l, 10.0), 0.50, 1.40)
        shading = cv2.GaussianBlur(shading, (7, 7), 1.5)
    else:
        shading = np.ones((h, w), dtype=np.float32)

    # 2. Modulate fabric with real wrinkles and room lighting
    shaded_cloth = (warped_rgb.astype(np.float32) * shading[:, :, None]).clip(0, 255).astype(np.uint8)

    # 3. Soft-edge feathered composite
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tight_mask = cv2.erode(cloth_mask, kernel_erode, iterations=1)
    mask_float = cv2.GaussianBlur(tight_mask.astype(np.float32) / 255.0, (7, 7), 0)[:, :, None]

    return (shaded_cloth * mask_float + frame_bgr * (1.0 - mask_float)).astype(np.uint8)

