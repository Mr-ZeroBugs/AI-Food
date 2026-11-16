"""Unified SegFormer-B2 Multi-Task 5-Food-Group inference engine (PyTorch CUDA & INT8 ONNX CPU)."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from food_taxonomy import extract_gz_to_temp


_MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax_np(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.maximum(np.sum(exp_v, axis=axis, keepdims=True), 1e-6)


class SegFormer5GroupEngine:
    """Single-model 10-class 5-Group segmentation + Food OOD gate + 5-Group soft ratio head."""

    def __init__(
        self,
        prefer_torch: bool = True,
        ckpt_path: Optional[str] = None,
        onnx_gz_path: Optional[str] = None,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ckpt_path = ckpt_path or os.path.join(base_dir, "models", "segformer_b2_5group_multitask.pt")
        self.onnx_gz_path = onnx_gz_path or os.path.join(base_dir, "models_onnx", "segformer_5group_int8.onnx.gz")
        self.backend = "unavailable"
        self.torch_model = None
        self.torch_device = None
        self.ort_sess = None

        if (
            prefer_torch
            and os.path.exists(self.ckpt_path)
            and os.environ.get("FORCE_VERCEL_ONNX") != "1"
            and os.environ.get("FORCE_LIGHTWEIGHT") != "1"
        ):
            try:
                import torch
                from train_5group_model import MultiTaskSegFormer5Group

                self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = MultiTaskSegFormer5Group()
                state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                self.torch_model = model.to(self.torch_device).eval()
                self.backend = f"segformer-b2-5group-torch-{self.torch_device.type}"
                return
            except Exception:
                self.torch_model = None

        if os.path.exists(self.onnx_gz_path) and os.environ.get("FORCE_LIGHTWEIGHT") != "1":
            import onnxruntime as ort

            cache_dir = os.path.join(tempfile.gettempdir(), "ai_food_onnx")
            dst = extract_gz_to_temp(
                self.onnx_gz_path, os.path.join(cache_dir, "segformer_5group_int8.onnx")
            )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_sess = ort.InferenceSession(
                dst, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.backend = "segformer-b2-5group-onnx"

    def predict(
        self, image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float, Dict[str, float]]:
        """
        Return:
          - local_mask: (H, W) uint8 in 0..9
          - mean_conf: float in [0..1]
          - latency_ms: float
          - food_prob: float in [0..1] (OOD non-food gate)
          - soft_ratios_pct: Dict[str, float] across the 5 food groups
        """
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        scale = 512.0 / max(w, h)
        nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        im_rs = np.asarray(pil_img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = ((im_rs - _MEAN_01) / _STD_01).transpose(2, 0, 1)
        if nh < 512 or nw < 512:
            pv_512 = np.pad(pv, ((0, 0), (0, 512 - nh), (0, 512 - nw)), mode="edge")[None, ...]
        else:
            pv_512 = pv[:, :512, :512][None, ...]

        t0 = time.perf_counter()
        if self.torch_model is not None and self.torch_device is not None:
            import torch

            if self.torch_device.type == "cuda":
                torch.cuda.synchronize()
            with torch.no_grad():
                seg_l, food_l, grp_r = self.torch_model(torch.from_numpy(pv_512).to(self.torch_device))
            if self.torch_device.type == "cuda":
                torch.cuda.synchronize()
            seg_np = seg_l[0].cpu().numpy()
            food_prob = float(torch.sigmoid(food_l[0, 0]).item())
            ratios_np = grp_r[0].cpu().numpy()
        elif self.ort_sess is not None:
            seg_l, food_l, grp_r = self.ort_sess.run(None, {"pixel_values": pv_512.astype(np.float32)})
            seg_np = seg_l[0]
            food_prob = float(1.0 / (1.0 + np.exp(-np.clip(float(food_l[0, 0]), -30.0, 30.0))))
            ratios_np = grp_r[0]
        else:
            raise RuntimeError("SegFormer5GroupEngine has no loaded PyTorch or ONNX weights")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ch, cw = max(1, int(round(nh / 4.0))), max(1, int(round(nw / 4.0)))
        seg_valid = seg_np[:, :ch, :cw]
        probs_valid = _softmax_np(seg_valid, axis=0)
        pred_small = np.argmax(probs_valid, axis=0).astype(np.uint8)
        conf_small = np.max(probs_valid, axis=0).astype(np.float32)

        local_mask = np.array(
            Image.fromarray(pred_small).resize((w, h), Image.NEAREST), dtype=np.uint8
        )
        # Filter tiny disconnected specks proportionally (< 0.1% of foreground pixels, capped)
        unique_ids, counts = np.unique(local_mask, return_counts=True)
        total_fg = int(np.sum(counts[unique_ids > 0]))
        min_px = min(250, max(20, int(0.001 * max(total_fg, 1))))
        for cid_np, px_np in zip(unique_ids, counts):
            if int(cid_np) > 0 and int(px_np) < min_px:
                local_mask[local_mask == int(cid_np)] = 0

        conf_map = np.asarray(
            Image.fromarray(conf_small).resize((w, h), Image.BILINEAR), dtype=np.float32
        )
        fg_mask = local_mask > 0
        mean_conf = (
            float(np.clip(conf_map[fg_mask].mean(), 0.0, 1.0))
            if np.any(fg_mask)
            else float(np.clip(conf_map.mean(), 0.0, 1.0))
        )

        from nutrition_engine import FIVE_GROUP_KEYS

        ratios_sum = max(float(np.sum(ratios_np)), 1e-6)
        soft_ratios_pct = {
            gk: round(float(ratios_np[i]) / ratios_sum * 100.0, 2)
            for i, gk in enumerate(FIVE_GROUP_KEYS)
        }
        return local_mask, mean_conf, latency_ms, food_prob, soft_ratios_pct
