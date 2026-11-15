"""FastAPI server for the AI 5-Food-Group Scanner & Real-Photo Benchmarks Dashboard."""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from real_foodseg103_engine import RealFoodSeg103Engine
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

app = FastAPI(title="AI Food 5-Groups Scanner", version="2.0.0")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

_STATE: Dict[str, object] = {"real_engine": None, "device": None}


def _get_real_engine() -> tuple[RealFoodSeg103Engine, torch.device]:
    if _STATE["real_engine"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _STATE["device"] = device
        _STATE["real_engine"] = RealFoodSeg103Engine(device=device)
    return _STATE["real_engine"], _STATE["device"]  # type: ignore[return-value]


def _load_preset_rgb(filename: str) -> np.ndarray:
    path = os.path.join(REAL_PRESETS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Real preset file missing: {filename}")
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail=f"Failed to read real preset: {filename}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _rgb_to_data_url(img_rgb: np.ndarray, max_dim: Optional[int] = None) -> str:
    work = img_rgb
    if max_dim is not None:
        h, w = work.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            work = cv2.resize(
                work,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    img_bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
    real_engine, device = _get_real_engine()
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
        except Exception as exc:
            raise HTTPException(status_code=400, detail="รูปแบบ Base64 ของรูปภาพไม่ถูกต้อง") from exc
        arr = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="ไม่สามารถอ่านไฟล์ภาพที่อัปโหลดได้")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dish_title = req.filename or "ภาพอาหารที่อัปโหลด"
    else:
        raise HTTPException(status_code=400, detail="กรุณาเลือกเมนูตัวอย่างหรืออัปโหลดภาพอาหาร")

    (
        pred_mask,
        _raw_foodseg_mask,
        mean_conf,
        latency_ms,
        _detailed_ingredients,
        thai_dish_info,
    ) = real_engine.predict(img_rgb)
    engine = f"5-Group Semantic AI ({device.type.upper()})"

    # Mild center-weighted relative relief for single-view 2.5D estimation
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
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

    uvicorn.run(app, host="127.0.0.1", port=8000)
