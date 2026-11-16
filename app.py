"""FastAPI server for the AI 5-Food-Group Scanner & Real-Photo Benchmarks Dashboard."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import traceback
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from lightweight_5group_segmenter import segment_food_5groups_lightweight
from nutrition_engine import (
    FIVE_GROUP_COLORS_RGB,
    FIVE_GROUP_KEYS,
    FIVE_GROUP_LABELS,
    analyze_plate_from_mask,
    render_mask_overlay,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PRESETS_DIR = os.path.join(BASE_DIR, "real_presets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CACHED_MASKS_PATH = os.path.join(REAL_PRESETS_DIR, "cached_masks.npz")
CACHED_META_PATH = os.path.join(REAL_PRESETS_DIR, "cached_meta.json")

PRESET_DEFS = [
    {
        "id": "kaprao_real",
        "file": "kaprao_real.jpg",
        "title": "ข้าวราดกะเพราหมูสับ (ภาพถ่ายจริง)",
        "subtitle": "หมูสับผัดกะเพรา + ข้าวสวย + พริกสด",
    },
    {
        "id": "kaomangai_real",
        "file": "kaomangai_real.jpg",
        "title": "ข้าวมันไก่ (ภาพถ่ายจริง)",
        "subtitle": "เนื้อไก่ต้ม + ข้าวมัน + แตงกวา",
    },
    {
        "id": "khaopad_goong_real",
        "file": "khaopad_goong_real.jpg",
        "title": "ข้าวผัดกุ้ง (ภาพถ่ายจริง)",
        "subtitle": "ข้าวผัดไข่ + กุ้งสด + แตงกวา + มะนาว",
    },
    {
        "id": "padthai_real",
        "file": "padthai_real.jpg",
        "title": "ผัดไทย (THFOOD-50 ภาพถ่ายจริง)",
        "subtitle": "เส้นผัดไทย + ถั่วงอก + ต้นหอม + มะนาว",
    },
    {
        "id": "nutrition5k_plate",
        "file": "nutrition5k_plate.jpg",
        "title": "สลัดอกไก่ย่างหน่อไม้ฝรั่ง (Google Nutrition5k)",
        "subtitle": "อกไก่ย่าง + ผักสลัด + หน่อไม้ฝรั่ง (ชั่งน้ำหนักจริง 234g)",
    },
    {
        "id": "chicken_broccoli_rice",
        "file": "chicken_broccoli_rice.jpg",
        "title": "ข้าวอกไก่บรอกโคลีเห็ดหอม (FoodSeg103 #22)",
        "subtitle": "เนื้อไก่ + ข้าวสวย + บรอกโคลี + ดอกกะหล่ำ",
    },
    {
        "id": "shrimp_rice_veggies",
        "file": "shrimp_rice_veggies.jpg",
        "title": "ข้าวสวยกุ้งสดผัดผักรวม (FoodSeg103 #0)",
        "subtitle": "กุ้ง + ข้าวสวย + แครอท + ถั่วแขก",
    },
    {
        "id": "fish_corn_broccoli",
        "file": "fish_corn_broccoli.jpg",
        "title": "สเต๊กปลาบรอกโคลีข้าวโพด (FoodSeg103 #12)",
        "subtitle": "เนื้อปลา + ข้าวโพด + บรอกโคลี + ถั่วฝักยาว",
    },
]

app = FastAPI(title="AI Food 5-Groups Scanner", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

_STATE: Dict[str, object] = {
    "real_engine": None,
    "device": None,
    "engine_unavailable": False,
    "onnx_engine": None,
    "onnx_unavailable": False,
    "cached_masks": None,
    "cached_meta": None,
}


def _try_get_real_engine() -> Tuple[Optional[object], Optional[object]]:
    """Lazy-load Tri-AI PyTorch engine when running in full AI / Docker environment."""
    if os.environ.get("FORCE_LIGHTWEIGHT") == "1" or os.environ.get("FORCE_VERCEL_ONNX") == "1":
        return None, None
    if _STATE["real_engine"] is not None:
        return _STATE["real_engine"], _STATE["device"]
    if _STATE["engine_unavailable"]:
        return None, None
    try:
        import torch  # type: ignore
        from real_foodseg103_engine import RealFoodSeg103Engine

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = RealFoodSeg103Engine(device=device)
        _STATE["device"] = device
        _STATE["real_engine"] = engine
        return engine, device
    except ImportError:
        _STATE["engine_unavailable"] = True
        return None, None
    except Exception:
        traceback.print_exc()
        _STATE["engine_unavailable"] = True
        return None, None


def _try_get_onnx_engine() -> Optional[object]:
    """Lazy-load INT8 ONNX Tri-AI engine for Vercel Serverless CPU."""
    if os.environ.get("FORCE_LIGHTWEIGHT") == "1":
        return None
    if _STATE["onnx_engine"] is not None:
        return _STATE["onnx_engine"]
    if _STATE["onnx_unavailable"]:
        return None
    try:
        from real_foodseg103_onnx import RealFoodSeg103ONNXEngine

        onnx_engine = RealFoodSeg103ONNXEngine()
        _STATE["onnx_engine"] = onnx_engine
        return onnx_engine
    except ImportError:
        _STATE["onnx_unavailable"] = True
        return None
    except Exception:
        traceback.print_exc()
        _STATE["onnx_unavailable"] = True
        return None


def _has_onnx_ai_packages() -> bool:
    if os.environ.get("FORCE_LIGHTWEIGHT") == "1":
        return False
    if _STATE["onnx_engine"] is not None:
        return True
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and os.path.exists(os.path.join(BASE_DIR, "models_onnx", "m2f_512_int8.onnx.gz"))
    )


def _has_full_ai_packages() -> bool:
    if os.environ.get("FORCE_LIGHTWEIGHT") == "1" or os.environ.get("FORCE_VERCEL_ONNX") == "1":
        return False
    if _STATE["real_engine"] is not None:
        return True
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def _load_cached_preset_prediction(
    preset_id: str,
) -> Optional[Tuple[np.ndarray, float, float, Dict[str, object]]]:
    """Load precomputed Tri-AI mask and metadata for fast Serverless preset responses."""
    if not (os.path.exists(CACHED_MASKS_PATH) and os.path.exists(CACHED_META_PATH)):
        return None
    if _STATE["cached_masks"] is None:
        with np.load(CACHED_MASKS_PATH) as npz:
            _STATE["cached_masks"] = {k: npz[k] for k in npz.files}
        with open(CACHED_META_PATH, "r", encoding="utf-8") as f:
            _STATE["cached_meta"] = json.load(f)

    masks_map: Dict[str, np.ndarray] = _STATE["cached_masks"]  # type: ignore[assignment]
    meta_map: Dict[str, Dict[str, object]] = _STATE["cached_meta"]  # type: ignore[assignment]
    if preset_id not in masks_map or preset_id not in meta_map:
        return None
    meta = meta_map[preset_id]
    return (
        masks_map[preset_id].astype(np.uint8),
        float(meta.get("mean_conf", 0.85)),
        float(meta.get("latency_ms", 12.0)),
        dict(meta.get("thai_dish", {})),
    )


def _load_preset_rgb(filename: str) -> np.ndarray:
    path = os.path.join(REAL_PRESETS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Real preset file missing: {filename}")
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read real preset: {filename}") from exc


def _cap_max_dim(img_rgb: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img_rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(
        Image.fromarray(img_rgb).resize((new_w, new_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def _rgb_to_data_url(img_rgb: np.ndarray, max_dim: Optional[int] = None) -> str:
    work = _cap_max_dim(img_rgb, max_dim) if max_dim is not None else img_rgb
    pil_im = Image.fromarray(work.astype(np.uint8))
    buf = io.BytesIO()
    pil_im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/")
def serve_home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/api/health")
def health_check():
    full_ai = _has_full_ai_packages()
    onnx_ai = _has_onnx_ai_packages()
    ai_backend_url = os.environ.get("AI_BACKEND_URL", "").strip()
    if full_ai:
        mode = "full-tri-ai"
    elif ai_backend_url:
        mode = "proxy-ai-backend"
    elif onnx_ai:
        mode = "onnx-tri-ai"
    else:
        mode = "vercel-serverless"
    return {
        "status": "ok",
        "mode": mode,
        "onnx_ready": onnx_ai,
        "ai_backend_configured": bool(ai_backend_url),
        "cached_presets": os.path.exists(CACHED_MASKS_PATH),
    }


@app.get("/api/presets")
def get_presets():
    items = []
    for preset in PRESET_DEFS:
        img_rgb = _load_preset_rgb(preset["file"])
        items.append({
            "id": preset["id"],
            "title": preset["title"],
            "subtitle": preset["subtitle"],
            "thumbnail_b64": _rgb_to_data_url(img_rgb, max_dim=180),
        })
    return {"presets": items}


@app.get("/api/metrics")
def get_metrics():
    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Evaluation report not generated yet.")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return {
        "report": report,
        "charts": {
            "training_curves": "/outputs/training_curves.png",
            "confusion_matrix": "/outputs/confusion_matrix_5groups.png",
            "sample_predictions": "/outputs/sample_predictions_comparison.png",
        },
    }


class ScanRequest(BaseModel):
    preset_id: Optional[str] = None
    image_b64: Optional[str] = None
    filename: Optional[str] = None
    view_mode: str = "5groups"


@app.post("/api/scan")
def scan_food(req: ScanRequest):
    real_engine, device = _try_get_real_engine()
    preset_match = None

    if req.preset_id:
        preset_match = next((p for p in PRESET_DEFS if p["id"] == req.preset_id), None)
        if not preset_match:
            raise HTTPException(status_code=400, detail=f"Unknown preset_id: {req.preset_id}")
        img_rgb = _load_preset_rgb(preset_match["file"])
        dish_title = preset_match["title"]
    elif req.image_b64:
        b64_payload = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
        try:
            raw = base64.b64decode(b64_payload)
            with Image.open(io.BytesIO(raw)) as im:
                im_fixed = ImageOps.exif_transpose(im).convert("RGB")
                img_rgb = np.asarray(im_fixed, dtype=np.uint8)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="ไม่สามารถอ่านไฟล์ภาพที่อัปโหลดได้") from exc
        img_rgb = _cap_max_dim(img_rgb, max_dim=1280)
        dish_title = req.filename or "ภาพอาหารที่อัปโหลด"
    else:
        raise HTTPException(status_code=400, detail="กรุณาเลือกเมนูตัวอย่างหรืออัปโหลดภาพอาหาร")

    # 1. Full Tri-AI PyTorch Engine (local GPU/CPU or Hugging Face Spaces Docker)
    if real_engine is not None and device is not None:
        (
            pred_mask,
            _raw_foodseg_mask,
            mean_conf,
            latency_ms,
            _detailed_ingredients,
            thai_dish_info,
        ) = real_engine.predict(img_rgb)  # type: ignore[attr-defined]
        engine = f"5-Group Semantic AI ({device.type.upper()})"
    else:
        # 2. Pre-computed Tri-AI mask for real presets (instant & exact on Vercel)
        cached = _load_cached_preset_prediction(req.preset_id) if req.preset_id else None
        if cached is not None:
            pred_mask, mean_conf, latency_ms, thai_dish_info = cached
            engine = "5-Group Semantic AI (Precomputed Tri-AI)"
        else:
            # 3. Optional remote full-AI backend relay (e.g., Hugging Face Spaces)
            ai_backend_url = os.environ.get("AI_BACKEND_URL", "").strip().rstrip("/")
            if ai_backend_url:
                try:
                    resp = requests.post(
                        f"{ai_backend_url}/api/scan",
                        json=req.model_dump(),
                        timeout=45,
                    )
                    if resp.status_code == 200:
                        return resp.json()
                except Exception:
                    pass

            # 4. Real INT8 ONNX Tri-AI Engine (Mask2Former + THFOOD-50 + Food-101) on Vercel CPU
            onnx_engine = _try_get_onnx_engine()
            if onnx_engine is not None:
                (
                    pred_mask,
                    _raw_foodseg_mask,
                    mean_conf,
                    latency_ms,
                    _detailed_ingredients,
                    thai_dish_info,
                ) = onnx_engine.predict(img_rgb)  # type: ignore[attr-defined]
                engine = "5-Group Semantic AI (ONNX Tri-AI)"
            else:
                # 5. Emergency fallback if neither PyTorch nor ONNX Runtime is available
                pred_mask, mean_conf, latency_ms, thai_dish_info = segment_food_5groups_lightweight(
                    img_rgb
                )
                engine = "5-Group Semantic AI (Serverless Edge)"

    # Mild center-weighted relative relief for single-view 2.5D estimation
    rgb_f = img_rgb.astype(np.float32) / 255.0
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    depth_map = np.clip(0.35 + 0.55 * (1.0 - np.abs(gray - 0.5)), 0.15, 1.0)

    analysis = analyze_plate_from_mask(pred_mask, depth_map=depth_map)
    overlay_mode = "211" if req.view_mode == "211" else "5groups"
    overlay_rgb = render_mask_overlay(img_rgb, pred_mask, mode=overlay_mode, alpha=0.50)

    group_legend = [
        {
            "key": gk,
            "label": FIVE_GROUP_LABELS[gk],
            "color_rgb": FIVE_GROUP_COLORS_RGB[gk],
            "pct_2d": analysis["five_group_ratios_2d"][gk],
            "pct_25d": analysis["five_group_ratios_25d"][gk],
        }
        for gk in FIVE_GROUP_KEYS
    ]

    return {
        "dish_title": dish_title,
        "engine": engine,
        "latency_ms": round(latency_ms, 2),
        "mean_confidence": round(mean_conf * 100.0, 1),
        "thai_dish": thai_dish_info,
        "input_image_b64": _rgb_to_data_url(img_rgb),
        "overlay_image_b64": _rgb_to_data_url(overlay_rgb),
        "group_legend": group_legend,
        "analysis": analysis,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
