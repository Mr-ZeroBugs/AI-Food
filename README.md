# AI-Food

AI-powered **5-Food-Group (อาหารหลัก 5 หมู่)** & **2:1:1 Healthy Plate (จานสุขภาพ 2:1:1)** nutrition scanner and segmentation dashboard built with FastAPI and a unified **Multi-Task SegFormer-B2 + Monocular Depth** pipeline.

Instead of getting bogged down by thousands of noisy fine-grained dish sub-classes, this system segments directly into the **5 Core Thai Food Groups** (`โปรตีน`, `คาร์โบไฮเดรต`, `เกลือแร่`, `วิตามิน`, `ไขมัน`), estimates **2.5D Depth-Weighted Volume/Mass** above a fitted plate plane, and rejects out-of-distribution non-food photos cleanly.

---

## Key Features

- **Unified Multi-Task 5-Group Segmentation (`segformer_5group_engine.py`):**
  - **Pixel Segmentation + Soft Ratios + Non-Food OOD Gate (`MultiTaskSegFormer5Group`):** Fine-tuned from `nvidia/segformer-b2-finetuned-ade-512-512` across `9,063` compiled real plates + surgical Thai egg refinement (`477` clean Thai egg dishes + stratified replay) from **FoodSeg103**, **Google Nutrition5k**, **THFOOD-50** (all 50 Thai menus), and strictly filtered **Tiny-ImageNet Non-Food** scenes.
  - **100% Non-Food OOD Rejection:** Learned binary food-vs-nonfood classification head paired with foreground pixel coverage & confidence guardrails (`100.0%` rejection on `n=100` unseen non-food scenes).
  - **Area-Proportional Noise Filtering (`food_taxonomy.py`):** Replaced fixed pixel thresholds with dynamic resolution-invariant filtering (`min(250, max(20, int(0.001 * total_food_px)))`).
- **2.5D Monocular Depth & Plate-Plane Elevation (`depth_volume_engine.py` + `nutrition_engine.py`):**
  - Replaces brightness heuristics with **`Depth-Anything-V2-Small`** (PyTorch GPU / INT8 ONNX CPU) + **2D Tilted Plate/Table Plane Subtraction (`fit_plate_plane_elevation`)** + **Nutrition5k-train-calibrated (`n=891`) 5-Group Density Weights**.
- **Dual Mask View Modes & Instant Vercel Serverless Parity:**
  - **Vercel Serverless Ready (`vercel.json` + `api/index.py`):** Packages only the lightweight quantized INT8 ONNX pair (`segformer_5group_int8.onnx.gz` `44.46 MB` + `depth_v2_s_int8.onnx.gz` `26.62 MB` = **`71.08 MB`**) inside Vercel's `500 MB` limit, delivering local/cloud parity across both presets and live uploads.
  - **Full AI Docker Backend (`Dockerfile` + `requirements-ai.txt`):** Ready for **Hugging Face Spaces (Docker)**, Railway, or Render on port `7860` to power both the Vercel frontend (`AI_BACKEND_URL`) and mobile apps (Flutter / React Native).

---

## Multi-Dataset Real-Photo Benchmark (`n = 300` Unseen Test Images)

Evaluated via `eval_real_foodseg103.py` on NVIDIA GeForce RTX 4060 (`26.33 ms` mean latency / `37.98 FPS`) and verified ONNX INT8 CPU parity (`benchmark_compare.py`):

| Benchmark Dataset | Samples (`n`) | Key Metrics |
| :--- | :---: | :--- |
| **EduardoPacheco/FoodSeg103** (Val) | 60 | Pixel Acc: **92.42%** (ONNX: **92.56%**) \| 5-Group mIoU: **76.21%** (ONNX: **76.47%**) \| MAE: **4.24% pts** \| 2:1:1 Acc: **90.0%** |
| **Google Research Nutrition5k** (Test) | 50 | 2.5D Depth+Density MAE: **10.77% pts** ($r = 0.631$, Blend MAE: **8.65% pts**, $r = 0.772$) \| Veg 2:1:1 Match: **82.0%** |
| **UEC-FoodPix Complete** (Val) | 40 | Pixel Acc: **92.47%** \| FG/BG mIoU: **85.09%** \| Food Mask IoU: **81.41%** |
| **Tiny-ImageNet Non-Food Holdout** | 100 | Learned OOD Gate Rejection Rate: **100.0%** (`100/100`) vs. baseline coverage gate `49.0%` |
| **THFOOD-50 Eval-500** (Ref Thai ViT) | 50 | Top-1 Thai Dish Acc: **90.0%** \| Top-3 Acc: **96.0%** \| SegFormer Food Recall: **100.0%** |

Generated evaluation plots and summary JSON are included in `outputs/`:
- `outputs/evaluation_report.json`
- `outputs/sample_predictions_comparison.png`
- `outputs/confusion_matrix_5groups.png`
- `outputs/training_curves.png`

---

## Deployment Options

### 1. Deploy Web App Immediately on Vercel
Import this GitHub repository in **Vercel** and click **Deploy**:
- Vercel uses `vercel.json` + `api/index.py` and installs `requirements.txt` (`fastapi`, `onnxruntime`, `numpy`, `pillow`, etc.) plus only `models_onnx/segformer_5group_int8.onnx.gz` and `models_onnx/depth_v2_s_int8.onnx.gz` (`71.08 MB` weights).
- Uploaded custom images execute real **INT8 ONNX SegFormer-B2 + Depth-Anything-V2-Small** directly on Vercel Serverless CPU, while all 8 presets use precomputed SegFormer-B2 masks (`real_presets/cached_masks.npz`) paired with live Depth-Anything elevation. If `AI_BACKEND_URL` is configured in Vercel Project Settings, uploads forward to your dedicated GPU/PyTorch server.

### 2. Deploy Dedicated PyTorch Backend on Hugging Face Spaces / Railway / Render
1. Create a new **Docker Space** on Hugging Face Spaces (or Railway/Render) connected to this repo.
2. It automatically builds `Dockerfile` with `requirements-ai.txt` and starts FastAPI on port `7860`.
3. Optionally set `AI_BACKEND_URL=https://<your-space>.hf.space` in Vercel Environment Variables.

---

## REST API Endpoints (CORS Enabled for Mobile App Integration)

- `GET /api/health` — Service status & active inference mode (`full-segformer-5group`, `onnx-segformer-5group`, `proxy-ai-backend`, or `vercel-serverless`)
- `GET /api/presets` — List of 8 real meal presets with thumbnails
- `GET /api/metrics` — Multi-dataset 300-image benchmark report & chart URLs
- `POST /api/scan` — Analyze a food image (`preset_id` or `image_b64`, `view_mode`: `"5groups"` or `"211"`) and return 5-food-group ratios, 2D vs. 2.5D depth/density breakdown, 2:1:1 balance score, recommendations, and segmentation mask overlay

---

## Local Development

1. **Full SegFormer-B2 CUDA/PyTorch mode:**
   ```bash
   pip install -r requirements-ai.txt
   python app.py
   ```
2. **Vercel INT8 ONNX CPU simulation:**
   ```bash
   pip install -r requirements.txt
   FORCE_VERCEL_ONNX=1 python app.py
   ```
   Open `http://127.0.0.1:8000` in your browser.
