# AI-Food

AI-powered **5-Food-Group (อาหารหลัก 5 หมู่)** & **2:1:1 Healthy Plate (จานสุขภาพ 2:1:1)** nutrition scanner and segmentation dashboard built with FastAPI and a pretrained Vision Transformer ensemble.

Instead of getting bogged down by thousands of noisy fine-grained dish sub-classes, this system aggregates pixel-level food segmentation directly into the **5 Core Thai Food Groups** (`โปรตีน`, `คาร์โบไฮเดรต`, `เกลือแร่`, `วิตามิน`, `ไขมัน`) alongside **2D Area** and **2.5D Volume/Mass** portion ratios and a **2:1:1 plate balance** check (`ผัก : ข้าว/แป้ง : โปรตีน`).

---

## Key Features

- **5-Food-Group Aggregation:** Groups segmentation outputs into the 5 foundational nutrition groups rather than fragile single-label dish names.
- **Dual Mask View Modes:** Toggle between **อาหารหลัก 5 หมู่ (5-Group Mask)** and **มุมมองจาน 2:1:1 (2:1:1 Plate View)** directly in the web UI.
- **2D Area vs. 2.5D Estimated Volume/Mass:** Compares pixel surface area ratios with single-view depth/density-weighted 2.5D estimates.
- **Tri-AI Pretrained Ensemble (`real_foodseg103_engine.py`):**
  - **Pixel Segmentation:** `Mask2Former Swin-T` (`mericberktas/foodseg-models`, `foodseg103/mask2former`)
  - **Thai Dish Context:** `SwinV2-Tiny` (`thean/swinv2-tiny-patch4-window8-256-finetuned-THFOOD-50`)
  - **Global Dish Context:** `Swin-Base` (`skylord/swin-finetuned-food101`)
- **Hybrid Cloud Architecture (Vercel + Docker AI Backend):**
  - **Vercel Serverless Ready (`vercel.json` + `api/index.py`):** Deploys in seconds under Vercel's size limits with precomputed Tri-AI preset caches (`real_presets/cached_masks.npz`), a built-in pure-NumPy Serverless Edge segmenter (`lightweight_5group_segmenter.py`), and optional `AI_BACKEND_URL` forwarding.
  - **Full AI Docker Backend (`Dockerfile` + `requirements-ai.txt`):** Ready for **Hugging Face Spaces (Docker)**, Railway, or Render on port `7860` to power both the Vercel frontend (`AI_BACKEND_URL`) and future Mobile Apps (Flutter / React Native) with CORS enabled.
- **Apple-Inspired Light Liquid Glass Web UI (`static/index.html`):** Drag-and-drop real photo upload plus 8 real-photo presets (`real_presets/`) and benchmark charts (`outputs/`).

---

## Multi-Dataset Real-Photo Benchmark (`n = 200` Test Images)

Evaluated via `eval_real_foodseg103.py` on NVIDIA GeForce RTX 4060 (`258.81 ms` mean latency / `3.86 FPS`) across 4 real food datasets (combined corpus size: `138,818` samples):

| Benchmark Dataset | Samples (`n`) | Key Metrics |
| :--- | :---: | :--- |
| **EduardoPacheco/FoodSeg103** (Val) | 60 | Pixel Acc: **87.54%** \| 5-Group mIoU: **67.99%** \| MAE: **7.07% pts** \| 2:1:1 Acc: **85.0%** |
| **Google Research Nutrition5k** (Test) | 50 | Gram Weight MAE: **11.05% pts** \| Correlation: **$r = 0.657$** \| 2:1:1 Veg Agreement: **78.0%** |
| **UEC-FoodPix Complete** (Val) | 40 | Pixel Acc: **90.28%** \| FG/BG mIoU: **80.92%** \| Food Mask IoU: **75.83%** |
| **THFOOD-50 Eval-500** | 50 | Top-1 Thai Dish Acc: **90.0%** \| Top-3 Acc: **96.0%** |

Generated evaluation plots and summary JSON are included in `outputs/`:
- `outputs/evaluation_report.json`
- `outputs/sample_predictions_comparison.png`
- `outputs/confusion_matrix_5groups.png`
- `outputs/training_curves.png`

---

## Deployment Options

### 1. Deploy Web App Immediately on Vercel
Import this GitHub repository in **Vercel** and click **Deploy**:
- Vercel uses `vercel.json` + `api/index.py` (`@vercel/python`) and installs lightweight `requirements.txt` (`< 70 MB` total bundle).
- All 8 real-photo presets use precomputed Tri-AI masks (`real_presets/cached_masks.npz`), while custom uploads run on `lightweight_5group_segmenter.py`—or forward automatically to your full PyTorch backend if you set the `AI_BACKEND_URL` environment variable in Vercel Project Settings (e.g., `https://your-space.hf.space`).

### 2. Deploy Full Tri-AI PyTorch Backend on Hugging Face Spaces / Railway / Render (for Web & Mobile App)
1. Create a new **Docker Space** on Hugging Face Spaces (or Railway/Render) connected to this repo.
2. It automatically builds `Dockerfile` with `requirements-ai.txt` and starts FastAPI on port `7860`.
3. Optionally set `AI_BACKEND_URL=https://<your-space>.hf.space` in Vercel Environment Variables so custom uploads on Vercel run the full 3-model PyTorch ensemble.

---

## REST API Endpoints (CORS Enabled for Mobile App Integration)

- `GET /api/health` — Service status & active inference mode (`full-tri-ai`, `proxy-ai-backend`, or `vercel-serverless`)
- `GET /api/presets` — List of 8 real meal presets with thumbnails
- `GET /api/metrics` — Multi-dataset 200-image benchmark report & chart URLs
- `POST /api/scan` — Analyze a food image (`preset_id` or `image_b64`, `view_mode`: `"5groups"` or `"211"`) and return 5-food-group ratios, 2D vs. 2.5D breakdown, 2:1:1 balance score, recommendations, and segmentation mask overlay

---

## Local Development

1. **Full Tri-AI mode (GPU/CPU):**
   ```bash
   pip install -r requirements-ai.txt
   python app.py
   ```
2. **Lightweight Vercel mode simulation:**
   ```bash
   pip install -r requirements.txt
   FORCE_LIGHTWEIGHT=1 python app.py
   ```
   Open `http://127.0.0.1:8000` in your browser.
