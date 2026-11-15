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

## Project Structure

```text
.
├── app.py                      # FastAPI backend & preset/analysis endpoints
├── real_foodseg103_engine.py   # Tri-AI pretrained ensemble & context fusion
├── nutrition_engine.py         # 5-food-group mapping, 2D/2.5D estimation & 2:1:1 evaluation
├── eval_real_foodseg103.py     # Multi-dataset real-photo benchmark runner
├── static/index.html           # Web UI dashboard
├── real_presets/               # Sample real meal photos for quick testing
├── outputs/                    # Benchmark report & visualization charts
└── requirements.txt            # Python dependencies
```

---

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI web app**

   ```bash
   python app.py
   ```

   Open `http://127.0.0.1:8000` in your browser. Pretrained Hugging Face weights download automatically on first inference.

3. **Run the 200-image multi-dataset benchmark (optional)**

   ```bash
   python eval_real_foodseg103.py
   ```

   `eval_real_foodseg103.py` automatically downloads and caches validation/test datasets into `datasets_cache/` (excluded from Git).
