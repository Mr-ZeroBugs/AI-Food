# AI-Food — Developer & AI Handoff Context

> **Purpose:** Comprehensive context for any AI assistant or developer resuming this project (especially when switching from Windows RTX 4060 development to **macOS / MacBook**). Everything below reflects verified repository state, architectural decisions, exact benchmark results, and the user's product vision.

---

## 1. User Goals & Product Vision

1. **Focus Strictly on the 5 Thai Food Groups (`อาหารหลัก 5 หมู่`) + `2:1:1` Healthy Plate:**
   - The user explicitly **does NOT want cluttered fine-grained dish classification** (hundreds/thousands of dish subclasses).
   - Output must aggregate cleanly into the **5 Core Food Groups**: `โปรตีน (protein)`, `คาร์โบไฮเดรต (carb)`, `เกลือแร่/ผัก (mineral_veg)`, `วิตามิน/ผักผลไม้ (vitamin_veg_fruit)`, and `ไขมัน (fat)`, plus the Thai **`2:1:1` ratio** (`ผัก 50% : ข้าว/แป้ง 25% : โปรตีน 25%`).
2. **Real-World Generalization (No Overfitting):**
   - Models must generalize to unseen restaurant/home-cooked plates, reject **non-food images** (dogs, objects, keyboards, empty tables, etc.), handle **Thai dishes** accurately (including `ไข่เจียว/ไข่ตุ๋น`, `ผัดไทย`, `กะเพรา`, `ข้าวมันไก่`, `ข้าวผัด`), and estimate volumetric/gram portions via 2.5D depth rather than raw 2D pixel area alone.
3. **Production Architecture (Vercel Web First -> Mobile App Next):**
   - **Web Prototype for Team Testing:** Deploy directly on **Vercel Serverless** via GitHub `main` branch (`https://github.com/Mr-ZeroBugs/AI-Food.git`) within Vercel's `< 500 MB` Python unzipped bundle limit.
   - **Mobile App Roadmap:** FastAPI backend (`app.py`) already has permissive CORS enabled and clean JSON REST endpoints (`/api/health`, `/api/presets`, `/api/metrics`, `/api/scan`) ready to power Flutter / React Native clients (or a dedicated Docker/HF Spaces GPU host via `Dockerfile` + `AI_BACKEND_URL`).
   - **Local vs. Vercel Parity:** Predictions between local dev machines and Vercel serverless CPU must match consistently.

---

## 2. Repository Architecture & What Was Built

### Core Production Inference Pipeline (Current `NEW v2` Default)

1. **`segformer_5group_engine.py` (`SegFormer5GroupEngine`) — Unified Multi-Task Vision Model**
   - Uses `MultiTaskSegFormer5Group` (defined in `train_5group_model.py`, fine-tuned from `nvidia/segformer-b2-finetuned-ade-512-512`) predicting **3 heads in one pass**:
     1. **10-Class -> 5-Group Pixel Mask (`seg_logits`):** `0=background, 1=rice, 2=noodles, 3=meat_cooked, 4=egg_fried, 5=egg_boiled, 6=leafy_greens, 7=root_veg, 8=fruit, 9=fried_rice`.
     2. **Learned Out-of-Distribution Food-vs-NonFood Gate (`food_logit`):** Binary probability (`food_prob`) rejecting non-food images before UI rendering.
     3. **Global Soft Portion Ratio Head (`group_ratios`):** Softmax prediction across the 5 food groups, supervised with 2x loss weight on laboratory-weighed plates.
   - Aspect-preserving `512x512` EdgePad preprocessing + adaptive speck noise filter: `min_px = min(250, max(20, int(0.001 * max(total_fg, 1))))`.
   - **Dual Backend Seamless Routing:** If `models/segformer_b2_5group_multitask.pt` + `torch` + `train_5group_model.py` exist (and `FORCE_VERCEL_ONNX != "1"`), uses PyTorch (`CUDA`/`CPU`). Otherwise (e.g., **on Vercel or fresh MacBook `git clone`**), automatically extracts and runs `models_onnx/segformer_5group_int8.onnx.gz` (`44.46 MB`) via `onnxruntime` CPU.

2. **`depth_volume_engine.py` (`DepthVolumeEngine`) — 2.5D Monocular Depth & Plate-Plane Subtraction**
   - Replaces legacy brightness-based pseudo-depth with **`Depth-Anything-V2-Small`** (`depth-anything/Depth-Anything-V2-Small-hf`, exported & quantized to `models_onnx/depth_v2_s_int8.onnx.gz`, `26.62 MB`, `~56.1 ms` CPU).
   - `fit_plate_plane_elevation(pred_depth, bg_mask)`: fits a least-squares 2D tilted plane `z = a*x + b*y + c` over subsampled non-food background/plate pixels and subtracts it from the depth map so tilted camera angles isolate actual food height above the plate.

3. **`nutrition_engine.py` — 5-Group Mapping, 2.5D Density Calibration & `2:1:1` Scoring**
   - Maps the 10 classes into `FIVE_GROUP_KEYS` via `FOOD_GROUP_WEIGHTS`.
   - Applies **`TRAIN_CALIBRATED_GROUP_DENSITY`** (`{'protein': 0.937, 'carb': 1.001, 'mineral_veg': 0.806, 'vitamin_veg_fruit': 1.195, 'fat': 0.72}`), calibrated strictly on `nutrition5k_train.parquet` (`n = 891` plates; zero test leakage), inside `analyze_plate_from_mask(..., apply_density=True)`.
   - Renders both `5groups` and `211` overlay masks via `render_mask_overlay`.

4. **`food_taxonomy.py` — Shared Taxonomy, Cache Safety & Thai Dish Rules**
   - `extract_gz_to_temp(src_gz, dst_path)`: decompresses `.onnx.gz` atomically to `/tmp/ai_food_onnx/` and checks `mtime` so updated `.gz` weights always re-extract instead of loading stale temp files.
   - Shared FoodSeg103 ID maps, area-proportional speck filtering, and `finalize_local_mask` rules (including promoting Thai egg dishes such as `KaiJeowMooSaap`, `KaiThoon`, `SonInLawEggs` out of false `fried_rice` mappings into `egg_fried`).

5. **`app.py`, `api/index.py`, `static/index.html`, `real_presets/`**
   - FastAPI server routing:
     - Preset scans when running Serverless/ONNX use freshly regenerated `real_presets/cached_masks.npz` + `cached_meta.json` (generated from the refined SegFormer-B2 checkpoint) + live Depth-Anything elevation.
     - Uploaded photos run `SegFormer5GroupEngine` + `DepthVolumeEngine` live (or `AI_BACKEND_URL` proxy if configured; fallback chain `RealFoodSeg103Engine` -> `RealFoodSeg103ONNXEngine` -> `segment_food_5groups_lightweight` stays intact for robustness).
   - **Non-Food Guardrail in `app.py`:** Triggers when `(food_prob < 0.50) or (food_cov < 0.015) or (mean_conf < 0.35 and food_cov < 0.05)`, resetting `health_score`, `five_group_ratios_*`, `plate_211`, `ingredients`, zeroing `display_mask`, and rendering `.advice-box.warning` in `static/index.html`.

6. **Deployment Config (`vercel.json`, `.vercelignore`, `Dockerfile`)**
   - `vercel.json` `includeFiles` packages ONLY `models_onnx/segformer_5group_int8.onnx.gz` (`44.46 MB`) and `models_onnx/depth_v2_s_int8.onnx.gz` (`26.62 MB`) (`74.28 MB` total `includeFiles` + `~153.89 MB` Python deps = **`~228.17 MB`**, safely under Vercel's `500 MB` unzipped limit).
   - Legacy ONNX files (`m2f_512_int8.onnx.gz`, `thfood50_int8.onnx.gz`, `food101_int8.onnx.gz`) remain tracked in Git for reference/comparison benchmarks (`benchmark_compare.py`) but are excluded from Vercel via `.vercelignore` and `vercel.json`.

---

## 3. Data Compilation & Training Pipeline (On Windows Dev Machine)

Because training datasets and PyTorch checkpoints are multi-GB, they are excluded via `.gitignore`. What was compiled & used:
- **`real_dataset_builder.py`**: Built `datasets_cache/compiled_5group_train.parquet` (`440.28 MB`, **`9,063` real samples**):
  - `FoodSeg103 Train`: `4,982` human-annotated plates
  - `Google Nutrition5k Train`: `888` lab-weighed plates (`has_gram_gt = 1.0`, weighted 2x in ratio loss)
  - `THFOOD-50 Train`: `2,393` Thai dishes sampled evenly across all 50 classes
  - `Tiny-ImageNet Non-Food`: `800` images sampled deterministically (`idx % 5 != 0` train split) from `167/200` strictly non-food WordNet classes (excluding food, plates, bowls, fruits, vegetables, cooked meats, etc.)
- **`train_5group_model.py`**: Trained `MultiTaskSegFormer5Group` for 6 epochs (`batch_size=4`, `grad_accum=3`, AdamW + CosineAnnealingLR, best checkpoint at Epoch 4: `Val mIoU = 73.66%`, `MAE = 4.59%`, `NonFoodReject = 100.0%`).
- **`refine_egg_segformer.py`**: Discovered that initial teacher pseudo-labeling had mapped Thai golden omelettes (`KaiJeowMooSaap`) into `fried_rice` (`egg_fried = 0` imgs). Fixed `food_taxonomy.py` and ran a 2-epoch Pareto-guarded surgical refinement using `477` cleaned Thai egg plates + stratified replay (`FoodSeg103 mIoU` rose `73.66% -> 73.72%`, unseen `THFOOD-50 Egg Holdout` rose **`30.8% -> 91.3%`**, `omelette_crop.jpg` switched to `egg_fried + rice`).
- Re-exported `models_onnx/segformer_5group_int8.onnx.gz` (`raw = 52.16 MB`, `gz = 44.46 MB`).

---

## 4. Measured Benchmarks (Verified Honest Numbers)

### Dashboard Benchmark (`outputs/evaluation_report.json` via `eval_real_foodseg103.py`, `n = 300`, RTX 4060)
- **FoodSeg103 (`n = 60`):** `mIoU = 76.21%` (`ONNX INT8: 76.47%`) | `PixAcc = 92.42%` (`ONNX: 92.56%`) | `MAE = 4.24%` | `2:1:1 Acc = 90.0%` | `26.1 ms` GPU (`505.5 ms` ONNX CPU)
- **Google Nutrition5k (`n = 50`):** `Gram MAE = 10.77%` | `Veg 2:1:1 Match = 82.0%` (Blend reaches `MAE 8.65%`, `r = 0.772` in `benchmark_compare.py`)
- **UEC-FoodPix (`n = 40`):** `Food Mask IoU = 81.41%` | `FG/BG mIoU = 85.09%` | `PixAcc = 92.47%`
- **Tiny-ImageNet Non-Food Holdout (`n = 100`):** `100.0%` (`100/100`) rejected
- **THFOOD-50 Ref Classifier (`n = 50`):** `Top-1 = 90.0%`, `Top-3 = 96.0%`, `SegFormer Food Recall = 100.0%`

### Large Untouched Head-to-Head A/B Evaluation (`n = 1,054` samples + `n = 20` CPU ONNX timing)
Comparing **OLD v1** (`RealFoodSeg103Engine` Mask2Former Tri-AI + brightness heuristic + no density + coverage gate) vs. **NEW v2** (`SegFormer5GroupEngine` + `DepthVolumeEngine` + `TRAIN_CALIBRATED_GROUP_DENSITY` + OOD gate):

| Evaluation Split | OLD v1 | NEW v2 | Notes |
| :--- | :--- | :--- | :--- |
| **Tiny-ImageNet Non-Food (`n = 300`, `idx % 5 == 0`)** | Reject **38.67%** (`116/300`) | Reject **100.0%** (`300/300`) | **+61.33%** OOD safety win |
| **Google Nutrition5k Full Test (`n = 100`)** | MAE **11.94%**, `r = 0.5706`, Veg **79.0%**, Recall **65.1%** | MAE **10.22%**, `r = 0.6638`, Veg **83.0%**, Recall **74.0%** *(+35% Soft-Ratio Blend: MAE 8.21%, r=0.7924)* | Decisive 2.5D portion accuracy win |
| **UEC-FoodPix Complete (`n = 250`)** | Food IoU **74.34%**, PixAcc **89.40%** | Food IoU **76.53%**, PixAcc **90.12%** | **+2.19%** Asian meal segmentation |
| **THFOOD-50 Egg Holdout (`n = 104`, `idx % 7 == 0`)** | **75.0%** *(even with new `food_taxonomy` rule)* | **91.3%** | **+16.3%** on Thai egg dishes |
| **FoodSeg103 Untouched `[120:420]` (`n = 300`)** | **mIoU 73.89%**, **PixAcc 90.73%**, 149.81 ms | mIoU **71.31%**, PixAcc **89.48%**, **28.37 ms** | **Honest tradeoff:** OLD Mask2Former still leads by `+2.58% mIoU` mainly on `Fruit/Vit` (`55.42%` vs `48.85%`) & `Carb` (`74.66%` vs `71.99%`), while `Protein` is tied (`66.29%` vs `66.03%`). |
| **Vercel CPU INT8 ONNX End-to-End (`n = 20`)** | **1,891.0 ms** \| Weights **144.01 MB** | **684.2 ms** (Seg + Depth) \| Weights **71.08 MB** | **2.76x faster CPU**, half bundle size |

---

## 5. What Was NOT Pushed to Git (Needed Only If Training/Fine-Tuning on MacBook)

Everything needed to run the **web app, REST API, and ONNX AI inference** IS in Git (`models_onnx/*.onnx.gz` + all Python/frontend code). The following `.gitignore`d items stay only on the original Windows PC:
- `models/segformer_b2_5group_multitask.pt` (`105.53 MB`) — copy this to MacBook `models/` if you want to resume PyTorch fine-tuning or run local PyTorch inference without re-training.
- `datasets_cache/` — `compiled_5group_train.parquet` (`440.28 MB`) plus raw parquets (`foodseg103_train_0..2`, `nutrition5k_train`, `thfood50_train_0..3`, etc.). Note: `eval_real_foodseg103.py` auto-downloads the validation datasets if missing, so only copy `datasets_cache/` if you want offline/training capability without rebuilding.
- Legacy unused files excluded by `.gitignore` (do not revive): `dataset_generator.py`, `model_seg.py`, `train_and_eval.py`, `yolo11_foodseg_template.py`.

---

## 6. Running & Continuing on MacBook

1. **Quick Web / API / App Development (`git clone` only — no extra files needed):**
   ```bash
   git clone https://github.com/Mr-ZeroBugs/AI-Food.git
   cd AI-Food
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   FORCE_VERCEL_ONNX=1 python app.py
   ```
   Open `http://127.0.0.1:8000` (runs `SegFormer-B2 INT8 ONNX + Depth-Anything-V2-Small INT8 ONNX` identically to Vercel).

2. **Suggested Future Improvements / Next Tasks:**
   - **Mobile App Frontend (Flutter / React Native / SwiftUI):** Consume `POST /api/scan` (`image_b64`, `view_mode: "5groups" | "211"`). Consider adding user feedback logging / opt-in misclassified image collection so the team can collect `50–200` real smartphone edge cases (street food, mixed rice-curry plates, low-light restaurants) for a future fine-tune.
   - **Closing the Remaining `Fruit/Vit` Gap (`48.85%` vs `55.42%`):** If higher accuracy on small fruits/western mixed salads is desired without sacrificing speed, increase sampling weight/loss weight on class `8 (fruit)` and `2 (noodles)` or distill further on FoodSeg103.
   - **Optional 2.5D Blend:** In `benchmark_compare.py`, blending `0.65 * mask_25d + 0.35 * soft_ratios` cut Nutrition5k gram MAE from `10.22% -> 8.21%` (`r = 0.7924`); currently `app.py` uses pure mask-derived 2.5D so percentages match painted pixel regions 1:1. Could be added carefully (e.g., weighted only over classes present in `pred_mask`).
   - **Environment Notes for Next AI:** Notice some scripts guard `sys.modules["torchvision"] = None` before importing `transformers` (workaround for a broken Windows `torchvision` wheel — harmless on macOS or can remain untouched). Always keep `vercel.json` `includeFiles` `< 500 MB` unzipped when adding files.
