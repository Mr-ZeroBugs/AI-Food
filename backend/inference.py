"""Server-side segmentation for clients that cannot run the model themselves (the mobile app and the
wearable camera). Same preprocessing and post-processing as the browser worker and training/evaluate.py,
so a photo gets the same answer wherever it is analysed.

Model files are chosen by environment variables (defaults: the fp32 export, which is ~2 points more
accurate than the int8 file the browser downloads):
  FOOD_MODEL_ONNX  path to the .onnx file
  FOOD_MODEL_META  path to its .json (classes, input size, stride, mean/std)
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_META = os.path.join(ROOT, "public", "models", "model.json")

# Overlay colours by plate group (same palette as the web app and the mobile app).
CLASS_RGB = {
    "rice": (240, 165, 0), "other_starch": (240, 165, 0), "sweets": (240, 165, 0),
    "animal_protein": (229, 72, 77), "egg": (229, 72, 77), "plant_protein": (229, 72, 77),
    "vegetable": (34, 160, 90), "fruit": (142, 78, 198),
    "mixed_dish": (176, 112, 40), "liquid": (80, 150, 210),
}
MAX_SIDE = 2048  # decode guard: larger photos are downscaled before anything else
SPECK = 0.003  # classes under 0.3% of the image become background (browser rule)


def _default_onnx(meta_path: str) -> str:
    meta = json.load(open(meta_path, encoding="utf-8"))
    fp32 = os.path.join(ROOT, "models", f"segformer_{meta.get('version', 'v3')}_fp32.onnx")
    return fp32 if os.path.exists(fp32) else os.path.join(os.path.dirname(meta_path), meta["file"])


class Segmenter:
    def __init__(self, onnx_path: str, meta_path: str, threads: int = 0) -> None:
        import onnxruntime as ort

        self.meta = json.load(open(meta_path, encoding="utf-8"))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.classes = self.meta["classes"]
        self.mean = np.array(self.meta["mean"], dtype=np.float32)
        self.std = np.array(self.meta["std"], dtype=np.float32)
        self.version = f"{self.meta.get('version', '?')}-{'fp32' if 'fp32' in os.path.basename(onnx_path) else 'int8'}"
        self._lock = threading.Lock()

    def run(self, im: Image.Image) -> Dict[str, object]:
        S, st = self.meta["input"], self.meta["output_stride"]
        w, h = im.size
        s = S / max(w, h)
        nw, nh = max(16, round(w * s)), max(16, round(h * s))
        arr = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = ((arr - self.mean) / self.std).transpose(2, 0, 1)
        pv = np.pad(pv, ((0, 0), (0, S - nh), (0, S - nw)), mode="edge")[None].astype(np.float32)
        with self._lock:  # one inference at a time per process keeps memory flat
            seg, food = self.sess.run(None, {"pixel_values": pv})
        ch, cw = max(1, round(nh / st)), max(1, round(nw / st))
        lab = seg[0][:, :ch, :cw].argmax(0).astype(np.uint8)
        for c in range(1, len(self.classes)):
            n = np.count_nonzero(lab == c)
            if 0 < n < SPECK * lab.size:
                lab[lab == c] = 0
        cnt = np.bincount(lab.ravel(), minlength=len(self.classes)) / lab.size
        return {
            "model_version": self.version,
            "class_areas": {self.classes[i]: float(v) for i, v in enumerate(cnt) if v > 0},
            "food_prob": float(1 / (1 + np.exp(-float(np.ravel(food)[0])))),
            "labels": lab,
        }

    def overlay_png_b64(self, lab: np.ndarray) -> str:
        """Transparent PNG of the class map (4x the model's output resolution); the client stretches it
        over the photo."""
        rgba = np.zeros((*lab.shape, 4), dtype=np.uint8)
        for i, name in enumerate(self.classes):
            col = CLASS_RGB.get(name)
            if col:
                rgba[lab == i] = (*col, 255)
        # 4x bilinear upscale: region edges come out soft instead of 4-pixel stair steps on the photo.
        im = Image.fromarray(rgba, "RGBA")
        im = im.resize((im.width * 4, im.height * 4), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


_segmenter: Optional[Segmenter] = None
_load_lock = threading.Lock()


def get_segmenter() -> Segmenter:
    global _segmenter
    if _segmenter is None:
        with _load_lock:
            if _segmenter is None:
                meta = os.environ.get("FOOD_MODEL_META", _DEFAULT_META)
                onnx_path = os.environ.get("FOOD_MODEL_ONNX") or _default_onnx(meta)
                _segmenter = Segmenter(onnx_path, meta, int(os.environ.get("FOOD_MODEL_THREADS", "0")))
    return _segmenter


def decode_image(data: bytes) -> Image.Image:
    """Open a JPEG/PNG/HEIC-converted upload, apply its EXIF rotation (phones store portrait photos
    sideways), and bound its size."""
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (MAX_SIDE, MAX_SIDE))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    return im
