import cv2
import numpy as np

def skeleton_warp_garment(garment_rgba, keypoints, target_h, target_w):
    """
    Warp garment semantic parts to pose keypoints:
    0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
    5: LShoulder, 6: LElbow, 7: LWrist,
    8: RHip, 9: RKnee, 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle
    """
    gh, gw = garment_rgba.shape[:2]

    kp = {}
    for i in range(18):
        if i < len(keypoints):
            kp[i] = np.array([keypoints[i][0] * target_w, keypoints[i][1] * target_h], dtype=np.float32)

    neck = kp.get(1, np.array([target_w * 0.45, target_h * 0.28]))
    r_sh = kp.get(2, np.array([target_w * 0.23, target_h * 0.31]))
    l_sh = kp.get(5, np.array([target_w * 0.62, target_h * 0.32]))
    r_elb = kp.get(3, np.array([target_w * 0.12, target_h * 0.48]))
    l_elb = kp.get(6, np.array([target_w * 0.69, target_h * 0.48]))
    r_wri = kp.get(4, np.array([target_w * 0.15, target_h * 0.63]))
    l_wri = kp.get(7, np.array([target_w * 0.64, target_h * 0.62]))
    r_hip = kp.get(8, np.array([target_w * 0.30, target_h * 0.66]))
    l_hip = kp.get(11, np.array([target_w * 0.55, target_h * 0.67]))
    r_knee = kp.get(9, np.array([target_w * 0.30, target_h * 0.92]))
    l_knee = kp.get(12, np.array([target_w * 0.58, target_h * 0.92]))

    sh_vec = l_sh - r_sh
    sh_width = np.linalg.norm(sh_vec) or 100.0
    sh_dir = sh_vec / sh_width
    perp = np.array([-sh_dir[1], sh_dir[0]]) # perpendicular vector pointing down
    if perp[1] < 0:
        perp = -perp

    # 1. TORSO QUAD
    # On Elaf cutout (1024x1536):
    src_torso = np.float32([
        [gw * 0.28, gh * 0.02],  # Top-right shoulder/collar
        [gw * 0.72, gh * 0.02],  # Top-left shoulder/collar
        [gw * 0.75, gh * 0.78],  # Bottom-left hem
        [gw * 0.25, gh * 0.78],  # Bottom-right hem
    ])
    
    # Destination reaches up to neck and across trapezius muscles
    dst_torso = np.float32([
        r_sh - perp * (sh_width * 0.22) - sh_dir * (sh_width * 0.15),
        l_sh - perp * (sh_width * 0.22) + sh_dir * (sh_width * 0.15),
        l_knee + sh_dir * (sh_width * 0.35),
        r_knee - sh_dir * (sh_width * 0.35),
    ])

    M_torso = cv2.getPerspectiveTransform(src_torso, dst_torso)
    warped_torso = cv2.warpPerspective(garment_rgba, M_torso, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # 2. RIGHT SLEEVE (Viewer's left = Person's right)
    src_r_sleeve = np.float32([
        [gw * 0.33, gh * 0.04],  # Top shoulder
        [gw * 0.35, gh * 0.36],  # Armpit / inner chest
        [gw * 0.24, gh * 0.78],  # Inner cuff
        [gw * 0.03, gh * 0.65],  # Outer cuff
    ])

    r_arm_vec = r_wri - r_sh
    r_arm_len = np.linalg.norm(r_arm_vec) or 100.0
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

    # 3. LEFT SLEEVE (Viewer's right = Person's left)
    src_l_sleeve = np.float32([
        [gw * 0.67, gh * 0.04],  # Top shoulder
        [gw * 0.65, gh * 0.36],  # Armpit / inner chest
        [gw * 0.76, gh * 0.78],  # Inner cuff
        [gw * 0.97, gh * 0.65],  # Outer cuff
    ])

    l_arm_vec = l_wri - l_sh
    l_arm_len = np.linalg.norm(l_arm_vec) or 100.0
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

    # 4. PANTS / TROUSERS (Lower hem to ankles)
    src_pants = np.float32([
        [gw * 0.30, gh * 0.78],
        [gw * 0.70, gh * 0.78],
        [gw * 0.70, gh * 0.99],
        [gw * 0.30, gh * 0.99],
    ])
    r_ank = kp.get(10, np.array([target_w * 0.31, target_h * 0.99]))
    l_ank = kp.get(13, np.array([target_w * 0.57, target_h * 0.99]))
    dst_pants = np.float32([
        r_knee - sh_dir * (sh_width * 0.2),
        l_knee + sh_dir * (sh_width * 0.2),
        l_ank + sh_dir * (sh_width * 0.2),
        r_ank - sh_dir * (sh_width * 0.2),
    ])
    M_pants = cv2.getPerspectiveTransform(src_pants, dst_pants)
    warped_pants = cv2.warpPerspective(garment_rgba, M_pants, (target_w, target_h), flags=cv2.INTER_LINEAR)

    # Blend layers (sleeves and pants under torso)
    def blend(base, layer):
        l_a = (layer[:, :, 3] / 255.0)[:, :, None]
        b_a = (base[:, :, 3] / 255.0)[:, :, None]
        out_rgb = layer[:, :, :3] * l_a + base[:, :, :3] * (1.0 - l_a) * b_a
        out_a = np.clip((l_a + b_a * (1.0 - l_a)) * 255.0, 0, 255).astype(np.uint8)[:, :, 0]
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
        return np.dstack([out_rgb, out_a])

    combined = blend(warped_pants, warped_r_sleeve)
    combined = blend(combined, warped_l_sleeve)
    combined = blend(combined, warped_torso)
    return combined

def realistic_tryon_blend(frame_bgr, warped_rgba, cloth_mask):
    """
    Apply physical shading, lighting, seamless infill, and clean clothing mask composite.
    """
    h, w = frame_bgr.shape[:2]
    warped_rgb = warped_rgba[:, :, :3].copy()
    garment_alpha = (warped_rgba[:, :, 3] / 255.0)

    # Fast 100% infill: Guarantee ZERO gaps inside cloth_mask
    mask_bool = (cloth_mask > 80)
    uncovered = mask_bool & (garment_alpha < 0.3)
    if np.any(uncovered):
        covered = (garment_alpha >= 0.5)
        if np.any(covered):
            # Sample median garment fabric color from valid garment pixels
            garment_pixels = warped_rgb[covered]
            fabric_color = np.median(garment_pixels, axis=0).astype(np.uint8)
            warped_rgb[uncovered] = fabric_color
            garment_alpha[uncovered] = 1.0

    # 1. EXTRACT LIGHTING & FOLD SHADING FIELD FROM WEARER'S VIDEO
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    cloth_l = l_channel[mask_bool]
    if len(cloth_l) > 0:
        mean_l = float(np.median(cloth_l))
        # Relative fold shading: highlights and shadows
        shading = l_channel / max(mean_l, 10.0)
        shading = np.clip(shading, 0.50, 1.40)
        shading = cv2.bilateralFilter(shading.astype(np.float32), 7, 0.25, 7)
    else:
        shading = np.ones((h, w), dtype=np.float32)

    # 2. MODULATE WARPED CLOTH WITH PHYSICAL SHADING FIELD
    shaded_cloth = (warped_rgb.astype(np.float32) * shading[:, :, None]).clip(0, 255).astype(np.uint8)

    # 3. SEAMLESS COMPOSITING WITH SOFT EDGE BLENDING
    # Erode dilated mask slightly to snap tightly to body silhouette without border spill
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tight_mask = cv2.erode(cloth_mask, kernel_erode, iterations=1)
    mask_float = cv2.GaussianBlur(tight_mask.astype(np.float32) / 255.0, (7, 7), 0)[:, :, None]
    result = (shaded_cloth * mask_float + frame_bgr * (1.0 - mask_float)).astype(np.uint8)
    return result

if __name__ == "__main__":
    from fashn_vton.dwpose import DWposeDetector

    print("Loading test frame 50 and garment...")
    frame = cv2.imread(".tmp_video_inspection/user_pose_50.jpg")
    h, w = frame.shape[:2]

    garment = cv2.imread("data/derived/product_cutouts/elaf_premium_10_front_cutout.png", cv2.IMREAD_UNCHANGED)
    cloth_mask = cv2.imread("data/derived/personal_dataset/masks/frame_0025.png", cv2.IMREAD_GRAYSCALE)
    if cloth_mask.shape[:2] != (h, w):
        cloth_mask = cv2.resize(cloth_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    print("Detecting body pose keypoints...")
    dw = DWposeDetector(checkpoints_dir="vendor/fashn-vton-1.5/weights/dwpose", device="cuda:0")
    pose = dw(frame)
    cand = pose["bodies"]["candidate"]

    print("Executing skeleton-guided garment warp...")
    warped = skeleton_warp_garment(garment, cand, h, w)

    print("Applying physical lighting, shading, and seamless composite...")
    tryon_result = realistic_tryon_blend(frame, warped, cloth_mask)

    cv2.imwrite("data/derived/test_realistic_tryon.jpg", tryon_result)
    print("Saved result to data/derived/test_realistic_tryon.jpg!")
