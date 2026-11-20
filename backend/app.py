"""HTTP API.

Endpoints
  GET  /api/health         -> which parts are available
  POST /api/analyze        -> web app: the browser segments, sends per-class areas (+ optional VLM)
  POST /api/analyze-image  -> mobile app / wearable: send the photo, segmentation runs on this server
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import vlm
from backend.nutrition import DAILY_TARGETS, build_result
from backend.ratelimit import client_key, limiter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(ROOT, "public")
MAX_IMAGE_B64_CHARS = 2_500_000  # ~1.8 MB JPEG; the browser sends ~150-400 KB

log = logging.getLogger("kinwhy")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # On the inference server, load the model before the first request instead of during it
    # (a cold Cloud Run instance would otherwise make its first user wait for the load).
    if os.environ.get("PRELOAD_MODEL") == "1":
        from backend import inference

        t0 = time.perf_counter()
        seg = inference.get_segmenter()
        log.info("model %s loaded in %.1fs", seg.version, time.perf_counter() - t0)
    yield


app = FastAPI(title="KinWhy AI", version="3.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Access-Code"],
)


@app.middleware("http")
async def isolation_headers(request, call_next):
    # Cross-origin isolation lets onnxruntime-web use multi-threaded WASM (same headers as vercel.json).
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return response


class SegSummary(BaseModel):
    model_version: str = "unknown"
    class_areas: Dict[str, float] = Field(default_factory=dict)
    food_prob: float = Field(1.0, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    image_b64: Optional[str] = None
    seg: SegSummary
    energy_level: int = 2000
    use_vlm: bool = True


def _check_access(code: Optional[str]) -> None:
    """Optional shared code so a public deployment can't be used to run up the API bill."""
    expected = os.environ.get("APP_ACCESS_CODE", "")
    if expected and not hmac.compare_digest(expected, code or ""):
        raise HTTPException(status_code=401, detail="ต้องใส่รหัสเข้าใช้งาน")


@app.get("/api/health")
def health():
    model = None
    if os.environ.get("PRELOAD_MODEL") == "1":
        from backend import inference

        model = inference.get_segmenter().version
    return {
        "status": "ok",
        "model": model,
        "version": app.version,
        "vlm_configured": vlm.is_configured(),
        "vlm_model": vlm.MODEL,
        "access_code_required": bool(os.environ.get("APP_ACCESS_CODE")),
        "energy_levels": sorted(DAILY_TARGETS),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, x_access_code: Optional[str] = Header(default=None)):
    _check_access(x_access_code)
    if req.energy_level not in DAILY_TARGETS:
        raise HTTPException(status_code=400, detail="energy_level ต้องเป็น 1600, 2000 หรือ 2400")
    for name, frac in req.seg.class_areas.items():
        if not (0.0 <= frac <= 1.0) or len(name) > 40:
            raise HTTPException(status_code=400, detail="ข้อมูลพื้นที่ไม่ถูกต้อง")

    vlm_result, vlm_error = None, None
    if req.use_vlm and req.image_b64:
        payload = req.image_b64.split(",", 1)[-1]
        if len(payload) > MAX_IMAGE_B64_CHARS:
            raise HTTPException(status_code=413, detail="ภาพใหญ่เกินไป")
        try:
            jpeg = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="อ่านไฟล์ภาพไม่ได้") from exc
        if not jpeg.startswith(b"\xff\xd8"):
            raise HTTPException(status_code=400, detail="ต้องเป็นภาพ JPEG")
        top = sorted(((k, v) for k, v in req.seg.class_areas.items() if k != "background" and v > 0.01),
                     key=lambda kv: -kv[1])[:6]
        hint = ", ".join(f"{k} {v * 100:.0f}% ของภาพ" for k, v in top) or None
        vlm_result, vlm_error = vlm.analyze_image(jpeg, seg_hint=hint)
    elif req.use_vlm:
        vlm_error = "ไม่ได้ส่งภาพมา"

    return build_result(req.seg.model_dump(), vlm_result, energy_level=req.energy_level, vlm_error=vlm_error)


MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@app.post("/api/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    energy_level: int = Form(2000),
    x_access_code: Optional[str] = Header(default=None),
):
    """Photo in, result out: for the mobile app and the wearable camera, which cannot run the model.
    Segmentation runs here (server-side ONNX); the response adds `overlay_png_b64`, a transparent
    colour map at the model's output resolution for the client to lay over the photo."""
    _check_access(x_access_code)
    if not limiter.allow(client_key(request.headers, request.client.host if request.client else None)):
        raise HTTPException(status_code=429, detail="ส่งรูปถี่เกินไป รอสักครู่แล้วลองใหม่")
    if energy_level not in DAILY_TARGETS:
        raise HTTPException(status_code=400, detail="energy_level ต้องเป็น 1600, 2000 หรือ 2400")
    t0 = time.perf_counter()
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ภาพใหญ่เกิน 8 MB")
    if not (data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG")):
        raise HTTPException(status_code=400, detail="ต้องเป็นภาพ JPEG หรือ PNG")

    from backend import inference  # onnxruntime is only needed on the inference server

    try:
        im = inference.decode_image(data)
    except Exception as exc:  # truncated or corrupt file
        raise HTTPException(status_code=400, detail="อ่านไฟล์ภาพไม่ได้") from exc
    seg_model = inference.get_segmenter()
    seg = await run_in_threadpool(seg_model.run, im)
    result = build_result(seg, None, energy_level=energy_level)
    result["warnings"] = [w for w in result.get("warnings", []) if "ประเมินปริมาณ" not in w]
    if result.get("is_food"):
        result["overlay_png_b64"] = seg_model.overlay_png_b64(seg["labels"])
    result["image_size"] = list(im.size)
    # Photos are never stored; the log keeps only sizes and timings.
    log.info("analyze-image %dx%d %dKB food=%s %.0fms", im.width, im.height, len(data) // 1024,
             result.get("is_food"), (time.perf_counter() - t0) * 1000)
    return result


# Local development: serve the web app from the same server. On Vercel the CDN serves public/, and the
# Cloud Run inference image sets SERVE_STATIC=0 (it ships no web app).
if os.path.isdir(PUBLIC_DIR) and not os.environ.get("VERCEL") and os.environ.get("SERVE_STATIC", "1") == "1":
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
