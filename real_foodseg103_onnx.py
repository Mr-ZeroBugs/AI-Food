"""Pure ONNX Runtime Tri-AI inference engine (zero PyTorch/Transformers dependency) for Vercel CPU."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from food_taxonomy import (
    FOOD101_THAI_NAMES,
    THFOOD50_THAI_NAMES,
    _UNSHIFT_LUT,
    finalize_local_mask,
    fuse_multimodel_context,
)


_MEAN_255 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD_255 = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _extract_gz_to_temp(models_dir: str, gz_name: str, cache_dir: str) -> str:
    """Decompress `.onnx.gz` once to ephemeral temp directory atomically."""
    raw_name = gz_name[:-3] if gz_name.endswith(".gz") else gz_name
    dst_path = os.path.join(cache_dir, raw_name)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
        return dst_path

    src_gz = os.path.join(models_dir, gz_name)
    if not os.path.exists(src_gz):
        raise FileNotFoundError(f"ONNX model asset missing: {src_gz}")

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{dst_path}.tmp.{os.getpid()}"
    with gzip.open(src_gz, "rb") as fin, open(tmp_path, "wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    os.replace(tmp_path, dst_path)
    return dst_path


class RealFoodSeg103ONNXEngine:
    """INT8 ONNX Tri-AI engine running Mask2Former + THFOOD-50 (+ lazy Food-101) on CPU."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = models_dir or os.path.join(base_dir, "models_onnx")
        self.cache_dir = os.path.join(tempfile.gettempdir(), "ai_food_onnx")

        labels_path = os.path.join(self.models_dir, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        self.thfood50_id2label: Dict[int, str] = {
            int(k): str(v) for k, v in labels["thfood50_id2label"].items()
        }
        self.food101_id2label: Dict[int, str] = {
            int(k): str(v) for k, v in labels["food101_id2label"].items()
        }

        self.sess_opts = ort.SessionOptions()
        self.sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        m2f_path = _extract_gz_to_temp(self.models_dir, "m2f_512_int8.onnx.gz", self.cache_dir)
        th_path = _extract_gz_to_temp(self.models_dir, "thfood50_int8.onnx.gz", self.cache_dir)
        self.m2f_sess = ort.InferenceSession(
            m2f_path, sess_options=self.sess_opts, providers=["CPUExecutionProvider"]
        )
        self.th_sess = ort.InferenceSession(
            th_path, sess_options=self.sess_opts, providers=["CPUExecutionProvider"]
        )
        self._food101_sess: Optional[ort.InferenceSession] = None

    def _get_food101_sess(self) -> Optional[ort.InferenceSession]:
        if self._food101_sess is not None:
            return self._food101_sess
        gz_path = os.path.join(self.models_dir, "food101_int8.onnx.gz")
        if not os.path.exists(gz_path):
            return None
        f101_path = _extract_gz_to_temp(self.models_dir, "food101_int8.onnx.gz", self.cache_dir)
        self._food101_sess = ort.InferenceSession(
            f101_path, sess_options=self.sess_opts, providers=["CPUExecutionProvider"]
        )
        return self._food101_sess

    @staticmethod
    def _run_cls_topk(
        sess: ort.InferenceSession,
        pil_img: Image.Image,
        size: int,
        id2label: Dict[int, str],
        name_map: Dict[str, str],
        k: int = 3,
    ) -> Dict[str, object]:
        arr = np.asarray(pil_img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = ((arr - _MEAN_01) / _STD_01).transpose(2, 0, 1)[None, ...]
        logits = sess.run(None, {"pixel_values": pv})[0][0]
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        top_idxs = np.argsort(probs)[::-1][:k]
        top3: List[Dict[str, object]] = []
        for idx in top_idxs:
            code = id2label[int(idx)]
            top3.append({
                "code": code,
                "thai_name": name_map.get(code, code.replace("_", " ")),
                "conf_pct": round(float(probs[int(idx)]) * 100.0, 1),
            })
        best = top3[0]
        return {
            "top1_code": best["code"],
            "top1_thai": best["thai_name"],
            "top1_conf_pct": best["conf_pct"],
            "top3": top3,
        }

    def predict(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, List[Dict[str, object]], Dict[str, object]]:
        """Run Tri-AI INT8 ONNX inference matching the PyTorch engine interface."""
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)

        t0 = time.perf_counter()

        # 1. Mask2Former 512x512 segmentation with aspect-preserving edge padding
        scale = 512.0 / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im_rs = np.asarray(pil_img.resize((nw, nh), Image.BILINEAR), dtype=np.float32)
        pv_rs = ((im_rs - _MEAN_255) / _STD_255).transpose(2, 0, 1)
        if nh < 512 or nw < 512:
            pv_m2f = np.pad(
                pv_rs, ((0, 0), (0, 512 - nh), (0, 512 - nw)), mode="edge"
            )[None, ...]
        else:
            pv_m2f = pv_rs[None, ...]
        ch = max(1, int(round(nh / 4.0)))
        cw = max(1, int(round(nw / 4.0)))

        class_logits, mask_logits = self.m2f_sess.run(None, {"pixel_values": pv_m2f})

        cq_exp = np.exp(class_logits - class_logits.max(axis=-1, keepdims=True))
        c_probs = (cq_exp / cq_exp.sum(axis=-1, keepdims=True))[..., :-1]
        m_probs = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -30.0, 30.0)))
        sem = np.einsum("bqc,bqhw->bchw", c_probs, m_probs)[0, :, :ch, :cw]

        # Fast top-2 per pixel via two argmax passes
        top1_idx = np.argmax(sem, axis=0)
        yy, xx = np.ogrid[: sem.shape[1], : sem.shape[2]]
        conf_128 = sem[top1_idx, yy, xx].astype(np.float32)
        sem_masked = sem.copy()
        sem_masked[top1_idx, yy, xx] = -1.0
        top2_idx = np.argmax(sem_masked, axis=0)

        r1 = _UNSHIFT_LUT[top1_idx.astype(np.uint8)].copy()
        r2 = _UNSHIFT_LUT[top2_idx.astype(np.uint8)]
        # Disambiguate generic 'other ingredients' (103) using top-2 concrete food class
        m103 = (r1 == 103) & (r2 > 0) & (r2 != 103)
        r1[m103] = r2[m103]

        raw_mask = np.array(Image.fromarray(r1).resize((w, h), Image.NEAREST), dtype=np.uint8)
        raw_top2_mask = np.array(Image.fromarray(r2).resize((w, h), Image.NEAREST), dtype=np.uint8)
        conf_map = np.asarray(
            Image.fromarray(conf_128).resize((w, h), Image.BILINEAR), dtype=np.float32
        )

        # 2. THFOOD-50 dish context + lazy Food-101 fallback when uncertain
        thai_dish_info = self._run_cls_topk(
            self.th_sess, pil_img, 256, self.thfood50_id2label, THFOOD50_THAI_NAMES
        )
        if float(thai_dish_info.get("top1_conf_pct", 0.0)) < 65.0:
            f101_sess = self._get_food101_sess()
            if f101_sess is not None:
                food101_info = self._run_cls_topk(
                    f101_sess, pil_img, 224, self.food101_id2label, FOOD101_THAI_NAMES
                )
            else:
                food101_info = {"top1_code": "", "top1_conf_pct": 0.0, "top3": []}
        else:
            food101_info = {"top1_code": "", "top1_conf_pct": 0.0, "top3": []}

        latency_ms = (time.perf_counter() - t0) * 1000.0

        thai_dish_info = fuse_multimodel_context(
            raw_mask, raw_top2_mask, thai_dish_info, food101_info
        )
        local_mask, mean_conf, detailed_ingredients = finalize_local_mask(
            raw_mask, conf_map, thai_dish_info
        )
        return local_mask, raw_mask, mean_conf, latency_ms, detailed_ingredients, thai_dish_info
