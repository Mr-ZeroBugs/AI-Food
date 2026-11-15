"""Lightweight pure-NumPy/PIL 5-food-group food segmenter for serverless CPU deployment."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def _box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast 2D box average over (H, W) float32 array using integral images."""
    h, w = arr.shape
    pad = np.pad(arr, ((radius, radius), (radius, radius)), mode="edge")
    cs = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
    cs[1:, 1:] = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    k = 2 * radius + 1
    total = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    return (total / float(k * k)).astype(np.float32)[:h, :w]


def _rgb_to_hsv_np(rgb_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB [0..1] -> Hue [0..360), Saturation [0..1], Value [0..1]."""
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    nz = delta > 1e-6
    mr = nz & (cmax == r)
    mg = nz & (cmax == g)
    mb = nz & (cmax == b)

    h[mr] = (60.0 * (((g[mr] - b[mr]) / delta[mr]) % 6.0))
    h[mg] = (60.0 * (((b[mg] - r[mg]) / delta[mg]) + 2.0))
    h[mb] = (60.0 * (((r[mb] - g[mb]) / delta[mb]) + 4.0))
    h = np.mod(h, 360.0)

    s = np.where(cmax > 1e-6, delta / np.maximum(cmax, 1e-6), 0.0).astype(np.float32)
    v = cmax.astype(np.float32)
    return h, s, v


def segment_food_5groups_lightweight(
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, float, float, Dict[str, object]]:
    """
    Segment an RGB plate photo into our 10 local nutrition classes (0..9)
    using only NumPy and Pillow so Vercel Serverless stays under 70 MB.
    """
    t0 = time.perf_counter()
    orig_h, orig_w = image_rgb.shape[:2]

    max_side = 224
    scale = min(1.0, max_side / max(orig_h, orig_w))
    work_w = max(32, int(round(orig_w * scale)))
    work_h = max(32, int(round(orig_h * scale)))

    pil_small = Image.fromarray(image_rgb).resize((work_w, work_h), resample=Image.BILINEAR)
    small_rgb = np.asarray(pil_small, dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv_np(small_rgb)

    r_ch, g_ch, b_ch = small_rgb[..., 0], small_rgb[..., 1], small_rgb[..., 2]
    gray = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
    local_mean_g = _box_mean(gray, radius=2)
    local_sq_g = _box_mean(gray * gray, radius=2)
    local_std = np.sqrt(np.maximum(0.0, local_sq_g - local_mean_g * local_mean_g))

    # Normalized elliptical center distance from plate center
    yy, xx = np.mgrid[0:work_h, 0:work_w]
    ny = (yy - work_h * 0.5) / max(work_h * 0.5, 1.0)
    nx = (xx - work_w * 0.5) / max(work_w * 0.5, 1.0)
    radial = np.sqrt(nx * nx + ny * ny)

    # Estimate outer corner/border color (table or outer plate rim)
    margin = max(4, min(work_h, work_w) // 12)
    border_mask = np.zeros((work_h, work_w), dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True
    border_rgb = np.median(small_rgb[border_mask], axis=0)
    border_dist = np.linalg.norm(small_rgb - border_rgb[None, None, :], axis=-1)

    # Candidate food likelihood vs background
    is_white_rim = (s < 0.13) & (v > 0.78) & (local_std < 0.045)
    is_dark_shadow = (v < 0.14) & (s < 0.35)
    is_border_like = (border_dist < 0.12) & (radial > 0.68) & (local_std < 0.05)

    # Score map for classes: [0: bg, 1: rice, 2: noodles, 3: meat_cooked, 4: egg_fried,
    #                         5: egg_boiled, 6: leafy_greens, 7: root_veg, 8: fruit, 9: fried_rice]
    scores = np.zeros((10, work_h, work_w), dtype=np.float32)

    center_prior = np.clip(1.25 - radial, 0.0, 1.0)
    bg_score = np.where(is_white_rim | is_dark_shadow | is_border_like, 1.35, 0.10)
    bg_score += np.clip((radial - 0.85) * 1.6, 0.0, 1.2)
    scores[0] = bg_score

    # 6: leafy_greens (emerald/green hues)
    green_hue = (h >= 62.0) & (h <= 165.0) & (s >= 0.18) & (v >= 0.14)
    scores[6] = np.where(green_hue, 1.45 + 0.6 * s, 0.0) * (0.5 + 0.5 * center_prior)

    # 7: root_veg (vibrant orange carrot, bright tomato red, lime-green cucumber skin, sweet corn)
    bright_orange = (h >= 12.0) & (h <= 34.0) & (s >= 0.55) & (v >= 0.45)
    fresh_red = ((h <= 10.0) | (h >= 344.0)) & (s >= 0.62) & (v >= 0.44)
    yellow_corn = (h >= 40.0) & (h <= 62.0) & (s >= 0.58) & (v >= 0.58)
    scores[7] = np.where(bright_orange | fresh_red | yellow_corn, 1.25 + 0.4 * s, 0.0)

    # 8: fruit (magenta/berry/purple/bright tropical pink)
    fruit_hue = (h >= 270.0) & (h <= 340.0) & (s >= 0.32) & (v >= 0.25)
    scores[8] = np.where(fruit_hue, 1.20 + 0.4 * s, 0.0)

    # 4: egg_fried (rich yolk / golden-white contrast)
    egg_yolk = (h >= 30.0) & (h <= 50.0) & (s >= 0.60) & (v >= 0.65)
    scores[4] = np.where(egg_yolk, 1.15 + 0.3 * s, 0.0)

    # 1: rice (textured bright white / off-white grains inside plate center)
    rice_like = (s <= 0.22) & (v >= 0.54) & (v <= 0.97) & (radial < 0.78)
    rice_tex = np.clip(local_std * 10.0, 0.0, 0.8)
    scores[1] = np.where(rice_like & (~is_white_rim), 0.75 + rice_tex + 0.45 * center_prior, 0.0)

    # 2: noodles (warm cream / golden carb strands)
    noodle_like = (h >= 30.0) & (h <= 55.0) & (s >= 0.20) & (s <= 0.54) & (v >= 0.52)
    scores[2] = np.where(noodle_like, 0.95 + np.clip(local_std * 6.0, 0.0, 0.4), 0.0)

    # 3: meat_cooked (warm browns, roasted meat, shrimp coral, poached chicken/pork)
    warm_brown_meat = ((h <= 38.0) | (h >= 332.0)) & (s >= 0.15) & (s <= 0.66) & (v >= 0.16) & (v <= 0.84)
    pale_meat = (s <= 0.24) & (v >= 0.32) & (v < 0.66) & ((r_ch - b_ch) > 0.03) & (radial < 0.76)
    scores[3] = np.where(warm_brown_meat | pale_meat, 1.05 + 0.35 * center_prior, 0.0)

    # Spatial smoothing across neighborhoods to eliminate pixel salt-and-pepper noise
    smooth_scores = np.stack([_box_mean(scores[c], radius=3) for c in range(10)], axis=0)
    small_mask = np.argmax(smooth_scores, axis=0).astype(np.uint8)

    # Remove tiny isolated class islands (< 0.5% of image area)
    total_px = work_h * work_w
    min_px = max(20, int(total_px * 0.005))
    for cid in range(1, 10):
        cnt = int(np.count_nonzero(small_mask == cid))
        if 0 < cnt < min_px:
            small_mask[small_mask == cid] = 0

    pred_mask = np.asarray(
        Image.fromarray(small_mask).resize((orig_w, orig_h), resample=Image.NEAREST),
        dtype=np.uint8,
    )

    # Approximate calibrated softmax confidence
    exp_s = np.exp(np.clip(smooth_scores - smooth_scores.max(axis=0, keepdims=True), -10.0, 0.0))
    probs = exp_s / np.maximum(exp_s.sum(axis=0, keepdims=True), 1e-6)
    max_probs = probs.max(axis=0)
    fg = small_mask > 0
    mean_conf = float(np.clip(max_probs[fg].mean(), 0.45, 0.92)) if np.any(fg) else 0.55

    latency_ms = (time.perf_counter() - t0) * 1000.0
    dish_info = {
        "top1_code": "5groups_serverless",
        "top1_thai": "โหมดวิเคราะห์ 5 หมู่ (Vercel Serverless)",
        "top1_conf_pct": round(mean_conf * 100.0, 1),
        "top3": [],
    }
    return pred_mask, mean_conf, latency_ms, dish_info
