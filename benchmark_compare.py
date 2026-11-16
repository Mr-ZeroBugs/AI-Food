"""Head-to-head real-data benchmark comparing Tri-AI Mask2Former vs Trained Multi-Task SegFormer-B2."""

from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from typing import Dict, List

if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from depth_volume_engine import DepthVolumeEngine
from food_taxonomy import FOODSEG103_ID2LABEL, FOODSEG103_TO_LOCAL_10CLASS
from nutrition_engine import CLASS_NAMES, FIVE_GROUP_KEYS, FOOD_GROUP_WEIGHTS, analyze_plate_from_mask
from real_foodseg103_engine import RealFoodSeg103Engine
from segformer_5group_engine import SegFormer5GroupEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "datasets_cache")
PRESETS_DIR = os.path.join(BASE_DIR, "real_presets")

_FS_LUT = np.zeros(256, dtype=np.uint8)
for _cid, _lcid in FOODSEG103_TO_LOCAL_10CLASS.items():
    if 0 <= _cid < 256:
        _FS_LUT[_cid] = _lcid

_NAME_TO_FSID = {name.lower(): cid for cid, name in FOODSEG103_ID2LABEL.items()}
_EVAL5_LUT = np.array([0, 2, 2, 1, 1, 1, 3, 3, 4, 2], dtype=np.int64)


def _eval_foodseg(predict_fn, n: int = 60) -> Dict[str, float]:
    rows = next(pq.ParquetFile(os.path.join(CACHE_DIR, "foodseg103_val.parquet")).iter_batches(batch_size=n)).to_pylist()
    cm = np.zeros((5, 5), dtype=np.int64)
    maes, lats, food_tp = [], [], 0
    for r in rows:
        im = Image.open(BytesIO(r["image"]["bytes"])).convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        gt = np.asarray(Image.open(BytesIO(r["label"]["bytes"])), dtype=np.uint8)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if gt.shape != arr.shape[:2]:
            gt = np.asarray(Image.fromarray(gt).resize((arr.shape[1], arr.shape[0]), Image.NEAREST), dtype=np.uint8)
        gt_local = _FS_LUT[gt]

        pred_local, lat_ms, food_prob = predict_fn(arr)
        lats.append(lat_ms)
        if food_prob >= 0.5:
            food_tp += 1

        cm += np.bincount(_EVAL5_LUT[gt_local].ravel() * 5 + _EVAL5_LUT[pred_local].ravel(), minlength=25).reshape(5, 5)
        gt_r = analyze_plate_from_mask(gt_local)["five_group_ratios_2d"]
        pr_r = analyze_plate_from_mask(pred_local)["five_group_ratios_2d"]
        maes.append(float(np.mean([abs(pr_r[k] - gt_r[k]) for k in FIVE_GROUP_KEYS])))

    diag = np.diag(cm).astype(np.float64)
    union = (cm.sum(axis=1) + cm.sum(axis=0) - diag).astype(np.float64)
    valid = union > 0
    return {
        "mIoU_pct": round(float(np.mean(diag[valid] / union[valid]) * 100.0), 2),
        "pixel_acc_pct": round(float(diag.sum() / max(cm.sum(), 1) * 100.0), 2),
        "MAE_pct_points": round(float(np.mean(maes)), 2),
        "food_recall_pct": round(food_tp / max(len(rows), 1) * 100.0, 1),
        "mean_lat_ms": round(float(np.mean(lats)), 1),
    }


def _eval_nutrition5k(predict_fn, depth_engine: DepthVolumeEngine, n: int = 50) -> Dict[str, float]:
    rows = next(pq.ParquetFile(os.path.join(CACHE_DIR, "nutrition5k_test.parquet")).iter_batches(batch_size=n)).to_pylist()
    maes_mask, maes_blend, gt_v, pr_m_v, pr_b_v = [], [], [], [], []
    for r in rows:
        arr = np.asarray(Image.open(BytesIO(r["image"]["bytes"])).convert("RGB"), dtype=np.uint8)
        gt_g = {k: 0.0 for k in FIVE_GROUP_KEYS}
        for fs_n, meta in zip(r["fs103_classes"], r["ingredients"]):
            cid = _NAME_TO_FSID.get(str(fs_n).lower(), 0)
            lcid = FOODSEG103_TO_LOCAL_10CLASS.get(cid, 0)
            if lcid <= 0:
                continue
            for gk, wt in FOOD_GROUP_WEIGHTS[CLASS_NAMES[lcid]].items():
                gt_g[gk] += max(float(meta["grams"]), 0.0) * wt
        tot = sum(gt_g.values())
        if tot <= 0:
            continue
        gt_r = {k: round(v / tot * 100.0, 2) for k, v in gt_g.items()}

        pred_local, soft_ratios = predict_fn(arr)
        elev = depth_engine.estimate_elevation_map(arr, pred_local)
        r_mask = analyze_plate_from_mask(pred_local, depth_map=elev, apply_density=True)["five_group_ratios_25d"]
        maes_mask.append(float(np.mean([abs(r_mask[k] - gt_r[k]) for k in FIVE_GROUP_KEYS])))
        if soft_ratios is not None:
            r_blend = {k: round(0.65 * float(r_mask[k]) + 0.35 * float(soft_ratios[k]), 2) for k in FIVE_GROUP_KEYS}
        else:
            r_blend = r_mask
        maes_blend.append(float(np.mean([abs(r_blend[k] - gt_r[k]) for k in FIVE_GROUP_KEYS])))
        for k in ("protein", "carb", "mineral_veg"):
            gt_v.append(gt_r[k])
            pr_m_v.append(float(r_mask[k]))
            pr_b_v.append(float(r_blend[k]))

    return {
        "gram_MAE_mask": round(float(np.mean(maes_mask)), 2),
        "gram_r_mask": round(float(np.corrcoef(gt_v, pr_m_v)[0, 1]), 3),
        "gram_MAE_blend": round(float(np.mean(maes_blend)), 2),
        "gram_r_blend": round(float(np.corrcoef(gt_v, pr_b_v)[0, 1]), 3),
    }


def _eval_nonfood(predict_fn, limit: int = 100) -> Dict[str, float]:
    nf_ids = set(json.load(open(os.path.join(CACHE_DIR, "tiny_nonfood_ids.json"), "r"))["non_food_ids"])
    rows = pq.ParquetFile(os.path.join(CACHE_DIR, "tiny_imagenet_val.parquet")).read_row_group(0).to_pylist()
    rejected, tested = 0, 0
    for idx, r in enumerate(rows):
        if idx % 5 != 0 or int(r["label"]) not in nf_ids:
            continue
        arr = np.asarray(Image.open(BytesIO(r["image"]["bytes"])).convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.uint8)
        pred_local, food_prob = predict_fn(arr)
        cov = np.count_nonzero(pred_local > 0) / max(pred_local.size, 1)
        if food_prob < 0.5 or cov < 0.015:
            rejected += 1
        tested += 1
        if tested >= limit:
            break
    return {"nonfood_reject_pct": round(rejected / max(tested, 1) * 100.0, 1), "n": tested}


def main() -> None:
    depth_engine = DepthVolumeEngine(prefer_torch=True)
    m2f = RealFoodSeg103Engine(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sf_torch = SegFormer5GroupEngine(prefer_torch=True)
    sf_onnx = SegFormer5GroupEngine(prefer_torch=False)

    print("=== FoodSeg103 Validation (n=60) ===")
    print("Mask2Former Tri-AI (CUDA) :", _eval_foodseg(lambda a: (lambda r: (r[0], r[3], 1.0))(m2f.predict(a))))
    print("SegFormer-B2 5Group (CUDA):", _eval_foodseg(lambda a: (lambda r: (r[0], r[2], r[3]))(sf_torch.predict(a))))
    print("SegFormer-B2 5Group (ONNX):", _eval_foodseg(lambda a: (lambda r: (r[0], r[2], r[3]))(sf_onnx.predict(a))))

    print("\n=== Google Nutrition5k Test Grams (n=50) ===")
    print("Mask2Former + Depth-Anything :", _eval_nutrition5k(lambda a: (m2f.predict(a)[0], None), depth_engine))
    print("SegFormer-B2 + Depth-Anything:", _eval_nutrition5k(lambda a: (lambda r: (r[0], r[4]))(sf_torch.predict(a)), depth_engine))

    print("\n=== Unseen Non-Food Holdout (n=100) ===")
    print("Mask2Former coverage check   :", _eval_nonfood(lambda a: (m2f.predict(a)[0], 1.0)))
    print("SegFormer-B2 OOD Gate        :", _eval_nonfood(lambda a: (lambda r: (r[0], r[3]))(sf_torch.predict(a))))


if __name__ == "__main__":
    main()
