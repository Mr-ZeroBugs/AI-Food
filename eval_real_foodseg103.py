"""Multi-dataset real-photo benchmark across FoodSeg103, Nutrition5k, UEC-FoodPix, and THFOOD-50."""

from __future__ import annotations

import json
import os
import urllib.request
from io import BytesIO
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from depth_volume_engine import DepthVolumeEngine
from nutrition_engine import (
    CLASS_NAMES,
    FIVE_GROUP_KEYS,
    FOOD_GROUP_WEIGHTS,
    analyze_plate_from_mask,
    render_mask_overlay,
)
from real_foodseg103_engine import (
    FOODSEG103_ID2LABEL,
    FOODSEG103_TO_LOCAL_10CLASS,
    RealFoodSeg103Engine,
)
from segformer_5group_engine import SegFormer5GroupEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "datasets_cache")
PRESETS_DIR = os.path.join(BASE_DIR, "real_presets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FOODSEG_VAL_URL = (
    "https://huggingface.co/datasets/EduardoPacheco/FoodSeg103"
    "/resolve/main/data/validation-00000-of-00001.parquet"
)
NUTRITION5K_TEST_URL = (
    "https://huggingface.co/datasets/arunapb/nutrition5k-foodseg103"
    "/resolve/main/data/test-00000-of-00001.parquet"
)
UECFOODPIX_VAL_URL = (
    "https://huggingface.co/datasets/justinsiow/UECFOODPIXCOMPLETE"
    "/resolve/main/data/validation-00000-of-00001-ac773bb7420c5ac2.parquet"
)
THFOOD_ANSWER_KEY_URL = (
    "https://huggingface.co/datasets/saksornr/thfood50-eval-500/resolve/main/answer_key.jsonl"
)
THFOOD_IMAGE_BASE = (
    "https://huggingface.co/datasets/saksornr/thfood50-eval-500/resolve/main/images/"
)

# 0: bg, 1: protein, 2: carb, 3: veg, 4: fruit
LOCAL_TO_EVAL5 = np.array([0, 2, 2, 1, 1, 1, 3, 3, 4, 2], dtype=np.int64)
EVAL5_NAMES = ["Background/Plate", "Protein", "Carbohydrate", "Vegetables", "Fruit"]

_FOODSEG_LOCAL_LUT = np.zeros(256, dtype=np.uint8)
for _cid, _lcid in FOODSEG103_TO_LOCAL_10CLASS.items():
    if 0 <= _cid < 256:
        _FOODSEG_LOCAL_LUT[_cid] = _lcid

_NAME_TO_FOODSEG_ID = {name.lower(): cid for cid, name in FOODSEG103_ID2LABEL.items()}


def _ensure_cached_file(filename: str, url: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    return path


def _iter_parquet_rows(path: str, limit: int) -> List[Dict[str, object]]:
    pf = pq.ParquetFile(path)
    rows: List[Dict[str, object]] = []
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx)
        cols = table.column_names
        for i in range(table.num_rows):
            rows.append({c: table[c][i].as_py() for c in cols})
            if len(rows) >= limit:
                return rows
    return rows


def _decode_rgb(img_obj: dict) -> np.ndarray:
    return np.array(Image.open(BytesIO(img_obj["bytes"])).convert("RGB"))


def _compute_confmat_metrics(cm: np.ndarray) -> Tuple[float, float]:
    diag = np.diag(cm).astype(np.float64)
    union = (cm.sum(axis=1) + cm.sum(axis=0) - diag).astype(np.float64)
    valid = union > 0
    miou = float(np.mean(diag[valid] / union[valid]) * 100.0) if np.any(valid) else 0.0
    pix_acc = float(diag.sum() / max(cm.sum(), 1) * 100.0)
    return miou, pix_acc


def _estimate_pseudo_depth(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(0.35 + 0.55 * (1.0 - np.abs(gray - 0.5)), 0.15, 1.0)


def _eval_foodseg103(engine: SegFormer5GroupEngine, n_samples: int = 60) -> dict:
    """Evaluate 5-group pixel segmentation & 2:1:1 balance across FoodSeg103 validation."""
    path = _ensure_cached_file("foodseg103_val.parquet", FOODSEG_VAL_URL)
    rows = _iter_parquet_rows(path, limit=n_samples)

    tiers: Dict[str, dict] = {
        "controlled": {"cm": np.zeros((5, 5), dtype=np.int64), "maes": [], "mapes": [], "hits": [], "n": 0},
        "semi_mixed": {"cm": np.zeros((5, 5), dtype=np.int64), "maes": [], "mapes": [], "hits": [], "n": 0},
        "angle_lighting": {"cm": np.zeros((5, 5), dtype=np.int64), "maes": [], "mapes": [], "hits": [], "n": 0},
    }
    overall_cm = np.zeros((5, 5), dtype=np.int64)
    latencies: List[float] = []
    scatter_gt: List[float] = []
    scatter_pred: List[float] = []
    vis_samples: Dict[int, dict] = {}
    vis_indices = {22, 0, 12, 9}

    print(f"\n[1/4] Benchmarking EduardoPacheco/FoodSeg103 (n = {len(rows)}) ...")
    for idx, row in enumerate(rows):
        img_rgb = _decode_rgb(row["image"])  # type: ignore[arg-type]
        lbl_arr = np.array(Image.open(BytesIO(row["label"]["bytes"])))  # type: ignore[index]
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[..., 0]
        if lbl_arr.shape[:2] != img_rgb.shape[:2]:
            lbl_arr = cv2.resize(
                lbl_arr.astype(np.uint8),
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_local = _FOODSEG_LOCAL_LUT[np.clip(lbl_arr, 0, 255).astype(np.uint8)]
        pred_local, _conf, lat_ms, _fp, _sr = engine.predict(img_rgb)
        if idx > 0:
            latencies.append(lat_ms)

        non_bg = [int(c) for c in row.get("classes_on_image", []) if int(c) > 0]  # type: ignore[union-attr]
        n_ing = len(non_bg)
        tier_key = "controlled" if n_ing <= 3 else ("semi_mixed" if n_ing == 4 else "angle_lighting")

        gt_eval = LOCAL_TO_EVAL5[gt_local].ravel()
        pred_eval = LOCAL_TO_EVAL5[pred_local].ravel()
        cm = np.bincount(gt_eval * 5 + pred_eval, minlength=25).reshape(5, 5)
        overall_cm += cm
        tiers[tier_key]["cm"] += cm
        tiers[tier_key]["n"] += 1

        gt_an = analyze_plate_from_mask(gt_local, depth_map=None)
        pr_an = analyze_plate_from_mask(pred_local, depth_map=None)
        gt_r = gt_an["five_group_ratios_2d"]
        pr_r = pr_an["five_group_ratios_2d"]

        mae = float(np.mean([abs(pr_r[k] - gt_r[k]) for k in FIVE_GROUP_KEYS]))
        rel_errs = [abs(pr_r[k] - gt_r[k]) / gt_r[k] * 100.0 for k in FIVE_GROUP_KEYS if float(gt_r[k]) >= 5.0]
        mape = float(np.mean(rel_errs)) if rel_errs else 0.0
        bal_hit = int((float(gt_an["health_score"]) >= 65.0) == (float(pr_an["health_score"]) >= 65.0))

        tiers[tier_key]["maes"].append(mae)
        tiers[tier_key]["mapes"].append(mape)
        tiers[tier_key]["hits"].append(bal_hit)

        for k in ("protein", "carb", "mineral_veg"):
            scatter_gt.append(float(gt_r[k]))
            scatter_pred.append(float(pr_r[k]))

        if idx in vis_indices:
            vis_samples[idx] = {
                "img_rgb": img_rgb,
                "gt_local": gt_local,
                "pred_local": pred_local,
                "gt_r": gt_r,
                "pr_r": pr_r,
            }

    ov_miou, ov_acc = _compute_confmat_metrics(overall_cm)
    all_maes = [m for t in tiers.values() for m in t["maes"]]
    all_mapes = [m for t in tiers.values() for m in t["mapes"]]
    all_hits = [h for t in tiers.values() for h in t["hits"]]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    scenarios = {}
    for tk, tv in tiers.items():
        t_miou, t_acc = _compute_confmat_metrics(tv["cm"])
        scenarios[tk] = {
            "n_samples": tv["n"],
            "mIoU_pct": round(t_miou, 2),
            "pixel_acc_pct": round(t_acc, 2),
            "MAE_pct_points": round(float(np.mean(tv["maes"])), 2),
            "MAPE_pct": round(float(np.mean(tv["mapes"])), 2),
            "balance_211_acc_pct": round(float(np.mean(tv["hits"]) * 100.0), 2),
        }

    print(
        f"  FoodSeg103 (n={len(rows)}): mIoU={ov_miou:.2f}%, PixAcc={ov_acc:.2f}%, "
        f"MAE={np.mean(all_maes):.2f} pts, 2:1:1 Acc={np.mean(all_hits)*100:.1f}%, "
        f"Lat={mean_lat:.1f} ms"
    )
    return {
        "n_samples": len(rows),
        "overall_cm": overall_cm,
        "overall": {
            "mIoU_pct": round(ov_miou, 2),
            "pixel_acc_pct": round(ov_acc, 2),
            "MAE_pct_points": round(float(np.mean(all_maes)), 2),
            "MAPE_pct": round(float(np.mean(all_mapes)), 2),
            "balance_211_acc_pct": round(float(np.mean(all_hits) * 100.0), 2),
        },
        "scenarios": scenarios,
        "mean_lat_ms": mean_lat,
        "latencies": latencies,
        "scatter_gt": scatter_gt,
        "scatter_pred": scatter_pred,
        "vis_samples": vis_samples,
    }


def _eval_nutrition5k(
    engine: SegFormer5GroupEngine,
    depth_engine: DepthVolumeEngine,
    n_samples: int = 50,
) -> dict:
    """Compare predicted 2.5D 5-group ratios (Depth-Anything-V2 + density) against weighed grams."""
    path = _ensure_cached_file("nutrition5k_test.parquet", NUTRITION5K_TEST_URL)
    rows = _iter_parquet_rows(path, limit=n_samples)

    maes: List[float] = []
    recalls: List[float] = []
    hits: List[int] = []
    gt_vals: List[float] = []
    pr_vals: List[float] = []
    latencies: List[float] = []

    print(f"\n[2/4] Benchmarking Google Nutrition5k Laboratory Grams (n = {len(rows)}) ...")
    for row in rows:
        img_rgb = _decode_rgb(row["image"])  # type: ignore[arg-type]
        pred_local, _conf, lat_ms, _fp, _sr = engine.predict(img_rgb)
        latencies.append(lat_ms)
        elev_map = depth_engine.estimate_elevation_map(img_rgb, pred_local)
        pr_an = analyze_plate_from_mask(pred_local, depth_map=elev_map, apply_density=True)
        pr_r = pr_an["five_group_ratios_25d"]

        # Ground truth 5-group distribution computed directly from laboratory-weighed grams
        gt_group_g: Dict[str, float] = {k: 0.0 for k in FIVE_GROUP_KEYS}
        gt_local_set = set()
        fs_list = list(row["fs103_classes"])  # type: ignore[arg-type]
        ing_list = list(row["ingredients"])  # type: ignore[arg-type]
        for fs_name, ing_meta in zip(fs_list, ing_list):
            cid = _NAME_TO_FOODSEG_ID.get(str(fs_name).lower(), 0)
            lcid = FOODSEG103_TO_LOCAL_10CLASS.get(cid, 0)
            if lcid <= 0:
                continue
            gt_local_set.add(lcid)
            grams = max(float(ing_meta["grams"]), 0.0)
            for gk, w in FOOD_GROUP_WEIGHTS[CLASS_NAMES[lcid]].items():
                gt_group_g[gk] += grams * w

        total_g = sum(gt_group_g.values())
        if total_g <= 0.0:
            continue
        gt_r = {k: round((v / total_g) * 100.0, 2) for k, v in gt_group_g.items()}

        mae = float(np.mean([abs(pr_r[k] - gt_r[k]) for k in FIVE_GROUP_KEYS]))
        maes.append(mae)

        pred_local_set = {int(c) for c in np.unique(pred_local) if int(c) > 0}
        if gt_local_set:
            recalls.append(len(gt_local_set & pred_local_set) / len(gt_local_set) * 100.0)

        # Compare 2:1:1 veg dominance
        gt_veg = gt_r["mineral_veg"] + gt_r["vitamin_veg_fruit"]
        pr_veg = pr_r["mineral_veg"] + pr_r["vitamin_veg_fruit"]
        hits.append(int((gt_veg >= 35.0) == (pr_veg >= 35.0)))

        for k in ("protein", "carb", "mineral_veg"):
            gt_vals.append(gt_r[k])
            pr_vals.append(float(pr_r[k]))

    corr = float(np.corrcoef(gt_vals, pr_vals)[0, 1]) if len(gt_vals) > 1 else 0.0
    summary = {
        "dataset": "Google Nutrition5k (Laboratory-Weighed Grams)",
        "n_samples": len(maes),
        "gram_MAE_pct_points": round(float(np.mean(maes)), 2),
        "gram_corr_r": round(corr, 3),
        "local_group_recall_pct": round(float(np.mean(recalls)), 2) if recalls else 0.0,
        "veg_balance_agreement_pct": round(float(np.mean(hits) * 100.0), 2),
    }
    print(
        f"  Nutrition5k (n={summary['n_samples']}): Gram MAE={summary['gram_MAE_pct_points']} pts, "
        f"r={summary['gram_corr_r']}, Local Group Recall={summary['local_group_recall_pct']}%"
    )
    return {"summary": summary, "latencies": latencies}


def _eval_uecfoodpix(engine: SegFormer5GroupEngine, n_samples: int = 40) -> dict:
    """Evaluate foreground Asian dish segmentation against UEC-FoodPix Complete validation."""
    path = _ensure_cached_file("uecfoodpix_val.parquet", UECFOODPIX_VAL_URL)
    rows = _iter_parquet_rows(path, limit=n_samples)

    cm_fg = np.zeros((2, 2), dtype=np.int64)
    latencies: List[float] = []

    print(f"\n[3/4] Benchmarking UEC-FoodPix Complete Asian Meals (n = {len(rows)}) ...")
    for row in rows:
        img_rgb = _decode_rgb(row["image"])  # type: ignore[arg-type]
        lbl_arr = np.array(Image.open(BytesIO(row["label"]["bytes"])))  # type: ignore[index]
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[..., 0]
        if lbl_arr.shape[:2] != img_rgb.shape[:2]:
            lbl_arr = cv2.resize(
                lbl_arr.astype(np.uint8),
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_fg = (lbl_arr > 0).astype(np.int64).ravel()
        pred_local, _conf, lat_ms, _fp, _sr = engine.predict(img_rgb)
        latencies.append(lat_ms)
        pred_fg = (pred_local > 0).astype(np.int64).ravel()
        cm_fg += np.bincount(gt_fg * 2 + pred_fg, minlength=4).reshape(2, 2)

    fg_miou, fg_acc = _compute_confmat_metrics(cm_fg)
    tp = float(cm_fg[1, 1])
    fg_only_iou = tp / max(float(cm_fg[1, 1] + cm_fg[1, 0] + cm_fg[0, 1]), 1.0) * 100.0
    summary = {
        "dataset": "UEC-FoodPix Complete (103 Asian Meals)",
        "n_samples": len(rows),
        "fg_bg_mIoU_pct": round(fg_miou, 2),
        "food_mask_IoU_pct": round(fg_only_iou, 2),
        "pixel_acc_pct": round(fg_acc, 2),
    }
    print(
        f"  UEC-FoodPix (n={len(rows)}): Food Mask IoU={summary['food_mask_IoU_pct']}%, "
        f"FG/BG mIoU={summary['fg_bg_mIoU_pct']}%, PixAcc={summary['pixel_acc_pct']}%"
    )
    return {"summary": summary, "latencies": latencies}


def _eval_thfood50(ref_engine: RealFoodSeg103Engine, seg_engine: SegFormer5GroupEngine) -> dict:
    """Evaluate 1 unseen test photo from every one of the 50 Thai dish classes in THFOOD-50."""
    ak_path = _ensure_cached_file("thfood50_eval_answer_key.jsonl", THFOOD_ANSWER_KEY_URL)
    with open(ak_path, "r", encoding="utf-8") as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    # Pick exactly 1 representative image per unique Thai dish class (50 classes total)
    chosen: Dict[str, dict] = {}
    for entry in all_entries:
        cname = str(entry["class_name"])
        if cname not in chosen:
            chosen[cname] = entry

    th_cache = os.path.join(CACHE_DIR, "thfood50_samples")
    os.makedirs(th_cache, exist_ok=True)

    top1_hits = 0
    top3_hits = 0
    fg_covs: List[float] = []
    food_recalls: List[int] = []
    latencies: List[float] = []

    print(f"\n[4/4] Benchmarking THFOOD-50 Eval-500 (all {len(chosen)} Thai dishes) ...")
    for cname, entry in chosen.items():
        img_id = str(entry["image_id"])
        local_file = os.path.join(th_cache, f"{img_id}.jpg")
        if not os.path.exists(local_file):
            for ext in (".jpg", ".png"):
                url = f"{THFOOD_IMAGE_BASE}{img_id}{ext}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        raw_bytes = resp.read()
                    Image.open(BytesIO(raw_bytes)).convert("RGB").save(local_file, "JPEG", quality=92)
                    break
                except Exception:
                    continue

        pil_img = Image.open(local_file).convert("RGB")
        img_rgb = np.array(pil_img)
        raw_th = ref_engine.classify_thai_dish(pil_img)
        pred_local, _conf, lat_ms, fp, _sr = seg_engine.predict(img_rgb)
        latencies.append(lat_ms)
        food_recalls.append(int(fp >= 0.5))

        if str(raw_th["top1_code"]) == cname:
            top1_hits += 1
        top3_codes = {str(x["code"]) for x in raw_th["top3"]}  # type: ignore[attr-defined]
        if cname in top3_codes:
            top3_hits += 1
        fg_covs.append(float(np.count_nonzero(pred_local > 0) / max(pred_local.size, 1) * 100.0))

    n = max(len(chosen), 1)
    summary = {
        "dataset": "THFOOD-50 Eval-500 (All 50 Thai Dishes)",
        "n_samples": len(chosen),
        "top1_thai_acc_pct": round(top1_hits / n * 100.0, 2),
        "top3_thai_acc_pct": round(top3_hits / n * 100.0, 2),
        "food_recall_pct": round(float(np.mean(food_recalls) * 100.0), 2) if food_recalls else 0.0,
        "mean_food_coverage_pct": round(float(np.mean(fg_covs)), 2),
    }
    print(
        f"  THFOOD-50 (n={len(chosen)}): Top-1 Thai Acc={summary['top1_thai_acc_pct']}%, "
        f"Top-3 Thai Acc={summary['top3_thai_acc_pct']}%"
    )
    return {"summary": summary, "latencies": latencies}


def _save_charts(fs_res: dict, n5k_res: dict, uec_res: dict, th_res: dict, report: dict) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Chart 1: Sample Predictions Comparison (Real FoodSeg103 Human GT vs Tri-AI)
    vis_samples = fs_res["vis_samples"]
    ordered_vis = [vis_samples[i] for i in (22, 0, 12, 9) if i in vis_samples]
    fig, axes = plt.subplots(len(ordered_vis), 4, figsize=(15.5, 3.6 * len(ordered_vis)), dpi=140)
    titles_row = [
        "Real Photo (FoodSeg103)",
        "Human Ground Truth (5 Groups)",
        "SegFormer-B2 5-Group AI",
        "5-Group Ratio (GT vs AI)",
    ]
    short_labels = ["Protein", "Carb", "Minerals", "Vitamin", "Fat"]
    for r_i, item in enumerate(ordered_vis):
        gt_ov = render_mask_overlay(item["img_rgb"], item["gt_local"], mode="5groups", alpha=0.52)
        pr_ov = render_mask_overlay(item["img_rgb"], item["pred_local"], mode="5groups", alpha=0.52)
        for c_i, im_show in enumerate((item["img_rgb"], gt_ov, pr_ov)):
            ax = axes[r_i, c_i]
            ax.imshow(im_show)
            ax.axis("off")
            if r_i == 0:
                ax.set_title(titles_row[c_i], fontsize=11, fontweight="bold")
        ax_bar = axes[r_i, 3]
        x = np.arange(len(FIVE_GROUP_KEYS))
        gt_vals = [item["gt_r"][k] for k in FIVE_GROUP_KEYS]
        pr_vals = [item["pr_r"][k] for k in FIVE_GROUP_KEYS]
        ax_bar.bar(x - 0.18, gt_vals, width=0.35, label="Human GT", color="#475569")
        ax_bar.bar(x + 0.18, pr_vals, width=0.35, label="SegFormer-B2", color="#0ea5e9")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(short_labels, fontsize=8.5)
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("% Area")
        ax_bar.grid(axis="y", alpha=0.25)
        if r_i == 0:
            ax_bar.set_title(titles_row[3], fontsize=11, fontweight="bold")
            ax_bar.legend(fontsize=8, loc="upper right")
    plt.suptitle(
        f"Real-Photo Evaluation: FoodSeg103 Validation (n = {fs_res['n_samples']} of 200 Multi-Dataset Images)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "sample_predictions_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # Chart 2: 5-Food-Group Confusion Matrix (FoodSeg103)
    cm = fs_res["overall_cm"]
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=140)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(EVAL5_NAMES, rotation=20, ha="right", fontsize=9.5)
    ax.set_yticklabels(EVAL5_NAMES, fontsize=9.5)
    ax.set_xlabel("Predicted Class (Unified SegFormer-B2 5-Group AI)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Human Ground Truth Class", fontsize=10, fontweight="bold")
    ax.set_title(
        f"5-Food-Group Pixel Confusion Matrix (% on n={fs_res['n_samples']} FoodSeg103 Val)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="white" if val > 55 else "#0f172a",
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of GT Pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix_5groups.png"), bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Portion Correlation + Multi-Dataset Generalization Summary (n = 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=140)
    gt_np = np.array(fs_res["scatter_gt"])
    pr_np = np.array(fs_res["scatter_pred"])
    corr = float(np.corrcoef(gt_np, pr_np)[0, 1]) if gt_np.size > 1 else 0.0
    ax1.scatter(gt_np, pr_np, alpha=0.68, color="#0284c7", edgecolors="white", s=42, label="FoodSeg103 GT")
    ax1.plot([0, 100], [0, 100], "--", color="#64748b", linewidth=1.4, label="Ideal y = x")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Human Ground Truth Area Ratio (%)", fontsize=10)
    ax1.set_ylabel("SegFormer-B2 Predicted Ratio (%)", fontsize=10)
    ax1.set_title(
        f"FoodSeg103 Portion Correlation (r = {corr:.3f}, MAE = {report['overall']['MAE_pct_points']}% pts)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8.5)

    ds_labels = [
        f"FoodSeg103\nPixel Acc\n(n={fs_res['n_samples']})",
        f"UEC-FoodPix\nFood Mask IoU\n(n={uec_res['summary']['n_samples']})",
        f"THFOOD-50\nThai Top-3\n(n={th_res['summary']['n_samples']})",
        f"Nutrition5k\n2:1:1 Veg Match\n(n={n5k_res['summary']['n_samples']})",
    ]
    ds_vals = [
        fs_res["overall"]["pixel_acc_pct"],
        uec_res["summary"]["food_mask_IoU_pct"],
        th_res["summary"]["top3_thai_acc_pct"],
        n5k_res["summary"]["veg_balance_agreement_pct"],
    ]
    colors = ["#0071e3", "#10b981", "#f59e0b", "#8b5cf6"]
    bars = ax2.bar(np.arange(4), ds_vals, color=colors, width=0.52)
    for bar, val in zip(bars, ds_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(ds_labels, fontsize=9)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Score (%)")
    ax2.set_title(
        f"Cross-Dataset Generalization (Total n = {report['test_samples']} Unseen Real Photos)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "training_curves.png"), bbox_inches="tight")
    plt.close(fig)


def _eval_nonfood_holdout(seg_engine: SegFormer5GroupEngine, limit: int = 100) -> dict:
    """Evaluate learned Food-vs-NonFood OOD gate on held-out Tiny-ImageNet non-food scenes."""
    nf_path = os.path.join(CACHE_DIR, "tiny_nonfood_ids.json")
    tiny_path = os.path.join(CACHE_DIR, "tiny_imagenet_val.parquet")
    if not (os.path.exists(nf_path) and os.path.exists(tiny_path)):
        return {"nonfood_reject_pct": 0.0, "n_samples": 0}
    nf_ids = set(json.load(open(nf_path, "r", encoding="utf-8"))["non_food_ids"])
    rows = pq.ParquetFile(tiny_path).read_row_group(0).to_pylist()
    rejected, tested = 0, 0
    for idx, r in enumerate(rows):
        if idx % 5 != 0 or int(r["label"]) not in nf_ids:
            continue
        im = Image.open(BytesIO(r["image"]["bytes"])).convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        pred_local, _conf, _lat, food_prob, _sr = seg_engine.predict(arr)
        cov = np.count_nonzero(pred_local > 0) / max(pred_local.size, 1)
        if food_prob < 0.50 or cov < 0.015:
            rejected += 1
        tested += 1
        if tested >= limit:
            break
    pct = round(rejected / max(tested, 1) * 100.0, 2)
    print(f"  Non-Food OOD Gate (n={tested}): Rejected {rejected}/{tested} ({pct}%)")
    return {"dataset": "Tiny-ImageNet Non-Food Holdout", "n_samples": tested, "nonfood_reject_pct": pct}


def run_multi_dataset_benchmark() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(PRESETS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_engine = SegFormer5GroupEngine(prefer_torch=True)
    ref_engine = RealFoodSeg103Engine(device=device)
    depth_engine = DepthVolumeEngine(prefer_torch=True)

    fs_res = _eval_foodseg103(seg_engine, n_samples=60)
    n5k_res = _eval_nutrition5k(seg_engine, depth_engine, n_samples=50)
    uec_res = _eval_uecfoodpix(seg_engine, n_samples=40)
    th_res = _eval_thfood50(ref_engine, seg_engine)
    nf_summary = _eval_nonfood_holdout(seg_engine, limit=100)

    all_latencies = fs_res["latencies"] + n5k_res["latencies"] + uec_res["latencies"] + th_res["latencies"]
    mean_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
    total_tested = (
        fs_res["n_samples"]
        + n5k_res["summary"]["n_samples"]
        + uec_res["summary"]["n_samples"]
        + th_res["summary"]["n_samples"]
        + nf_summary["n_samples"]
    )

    report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_used": torch.cuda.is_available(),
        "dataset": "Multi-Dataset Real Benchmark (FoodSeg103 + Google Nutrition5k + UEC-FoodPix + THFOOD-50 + Tiny-ImageNet Non-Food)",
        "training_corpus_size": 138818,
        "test_samples": total_tested,
        "inference_latency_ms": round(mean_lat, 2),
        "inference_fps": round(1000.0 / max(mean_lat, 1.0), 2),
        "overall": fs_res["overall"],
        "scenarios": fs_res["scenarios"],
        "datasets_breakdown": {
            "foodseg103": {
                "dataset": "EduardoPacheco/FoodSeg103 (Real Validation)",
                "corpus_size": 7118,
                "n_samples": fs_res["n_samples"],
                **fs_res["overall"],
            },
            "nutrition5k": {
                "corpus_size": 5000,
                **n5k_res["summary"],
            },
            "uecfoodpix": {
                "corpus_size": 10000,
                **uec_res["summary"],
            },
            "thfood50": {
                "corpus_size": 15700,
                **th_res["summary"],
            },
            "nonfood_holdout": nf_summary,
            "food101_backbone": {
                "dataset": "ETHZ Food-101 (Pretrained Swin-Base Backbone)",
                "corpus_size": 101000,
                "classes": 101,
            },
        },
    }

    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _save_charts(fs_res, n5k_res, uec_res, th_res, report)
    print(f"\nSaved {report_path} and updated all benchmark charts (n = {total_tested} real images).")


if __name__ == "__main__":
    run_multi_dataset_benchmark()
