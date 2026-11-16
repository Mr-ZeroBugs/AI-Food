"""Monocular Depth AI volume/elevation estimator (Depth-Anything-V2-Small) with 2D background plane fitting."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
from PIL import Image

from food_taxonomy import extract_gz_to_temp


_MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate_heuristic_depth(image_rgb: np.ndarray) -> np.ndarray:
    """Fast brightness-based fallback when Depth AI weights are unavailable."""
    rgb_f = image_rgb.astype(np.float32) / 255.0
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(0.35 + 0.55 * (1.0 - np.abs(gray - 0.5)), 0.15, 1.0).astype(np.float32)


def fit_plate_plane_elevation(pred_depth: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """
    Fit a 2D tilted background/table plane `z = a*x + b*y + c` on non-food pixels
    via least-squares and return relative food elevation above the plate surface.
    """
    h, w = pred_depth.shape
    step = max(1, min(h, w) // 48)
    d_sub = pred_depth[::step, ::step]
    b_sub = bg_mask[::step, ::step]
    sh, sw = d_sub.shape
    yy, xx = np.mgrid[0:sh, 0:sw]

    if np.count_nonzero(b_sub) >= 20:
        pts_x = (xx[b_sub] / max(sw - 1, 1)).astype(np.float64)
        pts_y = (yy[b_sub] / max(sh - 1, 1)).astype(np.float64)
        pts_z = d_sub[b_sub].astype(np.float64)
        design = np.column_stack([pts_x, pts_y, np.ones_like(pts_x)])
        coef, _, _, _ = np.linalg.lstsq(design, pts_z, rcond=None)
        gy = (np.arange(h)[:, None] / max(h - 1, 1)).astype(np.float32)
        gx = (np.arange(w)[None, :] / max(w - 1, 1)).astype(np.float32)
        plane = float(coef[0]) * gx + float(coef[1]) * gy + float(coef[2])
    else:
        plane = np.full_like(pred_depth, float(np.percentile(pred_depth, 10)))

    rel = np.maximum(0.0, pred_depth - plane)
    fg = ~bg_mask
    if np.any(fg):
        scale = max(float(np.percentile(rel[fg], 98)), 1e-3)
    else:
        scale = max(float(np.percentile(rel, 98)), 1e-3)
    return np.clip(0.25 + 1.40 * (rel / scale), 0.15, 2.0).astype(np.float32)


class DepthVolumeEngine:
    """Computes relative 3D food height via Depth-Anything-V2-Small (PyTorch CUDA or INT8 ONNX)."""

    def __init__(self, prefer_torch: bool = True, models_dir: Optional[str] = None) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = models_dir or os.path.join(base_dir, "models_onnx")
        self.backend = "heuristic"
        self.torch_model = None
        self.torch_proc = None
        self.torch_device = None
        self.ort_sess = None

        if prefer_torch and os.environ.get("FORCE_VERCEL_ONNX") != "1" and os.environ.get("FORCE_LIGHTWEIGHT") != "1":
            try:
                import sys
                if sys.modules.get("torchvision") is None:
                    sys.modules["torchvision"] = None  # type: ignore[assignment]
                import torch
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.torch_proc = AutoImageProcessor.from_pretrained(_DEPTH_REPO)
                self.torch_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_REPO).to(
                    self.torch_device
                )
                self.torch_model.eval()
                self.backend = f"depth-anything-v2-torch-{self.torch_device.type}"
                return
            except Exception:
                self.torch_model = None

        gz_path = os.path.join(self.models_dir, "depth_v2_s_int8.onnx.gz")
        if os.path.exists(gz_path) and os.environ.get("FORCE_LIGHTWEIGHT") != "1":
            try:
                import onnxruntime as ort

                cache_dir = os.path.join(tempfile.gettempdir(), "ai_food_onnx")
                dst = extract_gz_to_temp(gz_path, os.path.join(cache_dir, "depth_v2_s_int8.onnx"))
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_sess = ort.InferenceSession(
                    dst, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.backend = "depth-anything-v2-onnx"
            except Exception:
                self.ort_sess = None

    def estimate_elevation_map(self, image_rgb: np.ndarray, food_mask: np.ndarray) -> np.ndarray:
        """Predict relative height map aligned to `image_rgb` `(H, W)` above background plane."""
        h, w = image_rgb.shape[:2]
        bg_mask = food_mask == 0

        if self.torch_model is not None and self.torch_proc is not None:
            import torch

            pil_img = Image.fromarray(image_rgb)
            inputs = self.torch_proc(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                pred = self.torch_model(
                    pixel_values=inputs["pixel_values"].to(self.torch_device)
                ).predicted_depth
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
                )[0, 0].cpu().numpy()
            return fit_plate_plane_elevation(pred.astype(np.float32), bg_mask)

        if self.ort_sess is not None:
            pil_img = Image.fromarray(image_rgb).resize((280, 280), resample=Image.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            pv = ((arr - _MEAN_01) / _STD_01).transpose(2, 0, 1)[None, ...]
            pd = self.ort_sess.run(None, {"pixel_values": pv})[0][0]
            pred_d = np.asarray(
                Image.fromarray(pd.astype(np.float32)).resize((w, h), resample=Image.BILINEAR),
                dtype=np.float32,
            )
            return fit_plate_plane_elevation(pred_d, bg_mask)

        return estimate_heuristic_depth(image_rgb)
