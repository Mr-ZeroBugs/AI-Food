"""Evaluate a shipped ONNX segmenter (the exact file the app runs) on a fixed yardstick, so model
versions can be compared fairly.

    python training/evaluate.py public/models/segformer_v2_int8.onnx public/models/model_v2.json
    python training/evaluate.py public/models/segformer_v3_int8.onnx public/models/segformer_v3.json

Metrics
  FoodSeg103 val (2,135 human-labelled images, never trained on):
    plate_group_mIoU   IoU over background / protein / carb / veg / fruit, averaged. Pixels whose
                       ground truth is mixed dish / sweets / liquid are ignored here, so models with
                       and without those classes are compared on the same pixels.
    ratio_211_MAE      mean absolute error, in percentage points, of the veg:carb:protein plate
                       shares the app would show (computed with backend.nutrition.plate_area on the
                       predicted vs ground-truth masks). This is the number users actually see.
    class_mIoU         v3 11-class mIoU (only when the model uses the v3 classes).
  UECFoodPix val (1,000 Japanese meal photos): food-vs-background IoU.
  Tiny-ImageNet non-food (held-out idx % 5 == 0, 300 images): rejection rate with the app's rule.
  THFOOD-50 eval images (Thai dishes, no masks): share recognised as food with the app's rule.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.nutrition import MIN_FOOD_COVERAGE, plate_area  # noqa: E402
from backend.taxonomy import CLASSES_V3, CLASS_TO_GROUP_SPLIT, FOODSEG103_TO_V3  # noqa: E402

CACHE = os.path.join(ROOT, "datasets_cache")
EVAL_GROUPS = ["background", "protein", "carb", "veg", "fruit"]
_PRIMARY = {"g1_protein": 1, "g2_carb": 2, "g3_veg": 3, "g4_fruit": 4}
IGNORE = 255

FS_LUT = np.zeros(256, dtype=np.uint8)
for _k, _v in FOODSEG103_TO_V3.items():
    FS_LUT[_k] = _v


def class_to_eval_group(name: str) -> int:
    if name == "background":
        return 0
    if name in ("mixed_dish", "fried_rice", "sweets", "liquid"):
        return IGNORE
    split = CLASS_TO_GROUP_SPLIT[name]
    return _PRIMARY[max(split, key=split.get)]


V3_TO_EVAL = np.array([class_to_eval_group(n) for n in CLASSES_V3] + [IGNORE] * (256 - len(CLASSES_V3)), dtype=np.int64)


class Model:
    def __init__(self, onnx_path: str, meta_path: str, threads: int = 4) -> None:
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.classes: List[str] = self.meta["classes"]
        self.to_eval = np.array([class_to_eval_group(n) if n != "fried_rice" else IGNORE for n in self.classes], dtype=np.int64)
        self.mean = np.array(self.meta["mean"], dtype=np.float32)
        self.std = np.array(self.meta["std"], dtype=np.float32)
        self.ms: List[float] = []

    def predict(self, im: Image.Image):
        S = self.meta["input"]
        w, h = im.size
        s = S / max(w, h)
        nw, nh = max(16, round(w * s)), max(16, round(h * s))
        arr = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = ((arr - self.mean) / self.std).transpose(2, 0, 1)
        pv = np.pad(pv, ((0, 0), (0, S - nh), (0, S - nw)), mode="edge")[None].astype(np.float32)
        t0 = time.perf_counter()
        out = self.sess.run(None, {"pixel_values": pv})
        self.ms.append((time.perf_counter() - t0) * 1000)
        seg, food = out[0][0], out[1]
        st = self.meta["output_stride"]
        ch, cw = max(1, round(nh / st)), max(1, round(nw / st))
        lab = seg[:, :ch, :cw].argmax(0).astype(np.uint8)
        # Same speck rule as the browser worker: classes under 0.3% of the image -> background.
        n = lab.size
        for c in range(1, len(self.classes)):
            if 0 < np.count_nonzero(lab == c) < 0.003 * n:
                lab[lab == c] = 0
        food_prob = float(1 / (1 + np.exp(-float(food.ravel()[0]))))
        return lab, food_prob

    def class_areas(self, lab: np.ndarray) -> Dict[str, float]:
        cnt = np.bincount(lab.ravel(), minlength=len(self.classes)) / lab.size
        return {self.classes[i]: float(v) for i, v in enumerate(cnt) if v > 0}

    def is_food(self, lab: np.ndarray, food_prob: float) -> bool:
        cov = 1.0 - float(np.mean(lab == 0))
        return food_prob >= 0.5 and cov >= MIN_FOOD_COVERAGE


def _iou(cm: np.ndarray) -> np.ndarray:
    d = np.diag(cm).astype(float)
    u = cm.sum(0) + cm.sum(1) - d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(u > 0, d / u, np.nan)


def eval_foodseg(m: Model, limit: int | None) -> Dict[str, object]:
    cm = np.zeros((5, 6), dtype=np.int64)
    ncls = len(CLASSES_V3)
    cm_cls = np.zeros((ncls, ncls), dtype=np.int64)
    use_cls = m.classes == CLASSES_V3
    maes: List[float] = []
    n = 0
    for batch in pq.ParquetFile(os.path.join(CACHE, "foodseg103_val.parquet")).iter_batches(batch_size=32, columns=["image", "label"]):
        for r in batch.to_pylist():
            if limit and n >= limit:
                break
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            gt_raw = np.asarray(Image.open(io.BytesIO(r["label"]["bytes"])), dtype=np.uint8)
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[..., 0]
            lab, _ = m.predict(im)
            # Compare at the model's output resolution (what the app uses for areas).
            gt_v3 = np.asarray(Image.fromarray(FS_LUT[gt_raw]).resize(lab.shape[::-1], Image.NEAREST), dtype=np.uint8)
            g = V3_TO_EVAL[gt_v3].ravel()
            p = m.to_eval[lab].ravel()
            keep = g != IGNORE
            # Column 5 = "predicted mixed/sweets/liquid" on a pure-group pixel: a miss for the true
            # group (adds to its union) without inflating any other group's union.
            p = np.where(p == IGNORE, 5, p)
            cm += np.bincount(g[keep] * 6 + p[keep], minlength=30).reshape(5, 6)
            if use_cls:
                gg, pp = gt_v3.ravel().astype(np.int64), lab.ravel().astype(np.int64)
                cm_cls += np.bincount(gg * ncls + pp, minlength=ncls * ncls).reshape(ncls, ncls)

            gt_area = {CLASSES_V3[i]: float(v) for i, v in enumerate(np.bincount(gt_v3.ravel(), minlength=ncls) / gt_v3.size) if v > 0}
            a_gt = plate_area(gt_area)["plate_211"]
            a_pr = plate_area(m.class_areas(lab))["plate_211"]
            if a_gt["score"] is not None:
                if a_pr["score"] is None:
                    maes.append(100.0 / 3 * 2)  # predicted nothing on a real plate: count as large error
                else:
                    maes.append(float(np.mean([abs(a_gt["ratio"][k] - a_pr["ratio"][k]) for k in ("veg", "carb", "protein")])))
            n += 1
        if limit and n >= limit:
            break
    d = np.diag(cm[:, :5]).astype(float)
    union = cm.sum(1) + cm[:, :5].sum(0) - d
    iou = np.where(union > 0, d / np.maximum(union, 1), np.nan)
    out: Dict[str, object] = {
        "n": n,
        "plate_group_mIoU": round(float(np.nanmean(iou)) * 100, 2),
        "plate_group_IoU": {EVAL_GROUPS[i]: round(float(iou[i]) * 100, 2) for i in range(5)},
        "ratio_211_MAE_pp": round(float(np.mean(maes)), 2),
        "ratio_211_MAE_median_pp": round(float(np.median(maes)), 2),
    }
    if use_cls:
        ci = _iou(cm_cls)
        out["class_mIoU"] = round(float(np.nanmean(ci)) * 100, 2)
        out["class_IoU"] = {CLASSES_V3[i]: (None if np.isnan(ci[i]) else round(float(ci[i]) * 100, 2)) for i in range(ncls)}
    return out


def eval_thai(m: Model) -> Dict[str, object]:
    """Thai test set (training/sam_label.py): SAM-drawn regions, every region checked by a reviewer.
    Pixels marked uncertain (255) are ignored. Images never used in any training run."""
    path = os.path.join(CACHE, "thai_sam_test.parquet")
    if not os.path.exists(path):
        return {}
    ncls = len(CLASSES_V3)
    cm = np.zeros((5, 6), dtype=np.int64)
    cm_cls = np.zeros((ncls, ncls), dtype=np.int64)
    use_cls = m.classes == CLASSES_V3
    maes: List[float] = []
    per_dish: Dict[str, List[float]] = {}
    for r in pq.read_table(path).to_pylist():
        im = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
        gt_full = np.asarray(Image.open(io.BytesIO(r["mask_bytes"])), dtype=np.uint8)
        lab, _ = m.predict(im)
        gt = np.asarray(Image.fromarray(gt_full).resize(lab.shape[::-1], Image.NEAREST), dtype=np.uint8)
        valid = gt != IGNORE
        g = np.where(valid, V3_TO_EVAL[np.minimum(gt, 255)], IGNORE).ravel()
        p = m.to_eval[lab].ravel()
        keep = g != IGNORE
        p = np.where(p == IGNORE, 5, p)
        cm += np.bincount(g[keep] * 6 + p[keep], minlength=30).reshape(5, 6)
        if use_cls:
            gg, pp = gt.ravel()[valid.ravel()].astype(np.int64), lab.ravel()[valid.ravel()].astype(np.int64)
            cm_cls += np.bincount(gg * ncls + pp, minlength=ncls * ncls).reshape(ncls, ncls)
        # 2:1:1 on the reviewed pixels only (uncertain pixels removed from both sides).
        gt_area = {CLASSES_V3[i]: float(v) for i, v in enumerate(np.bincount(gt[valid], minlength=ncls) / max(valid.sum(), 1)) if v > 0}
        pl = lab[valid]
        pr_area = {m.classes[i]: float(v) for i, v in enumerate(np.bincount(pl, minlength=len(m.classes)) / max(pl.size, 1)) if v > 0}
        a_gt, a_pr = plate_area(gt_area)["plate_211"], plate_area(pr_area)["plate_211"]
        if a_gt["score"] is not None:
            e = 100.0 / 3 * 2 if a_pr["score"] is None else float(np.mean([abs(a_gt["ratio"][k] - a_pr["ratio"][k]) for k in ("veg", "carb", "protein")]))
            maes.append(e)
            per_dish.setdefault(r["dish"], []).append(e)
    d = np.diag(cm[:, :5]).astype(float)
    union = cm.sum(1) + cm[:, :5].sum(0) - d
    iou = np.where(union > 0, d / np.maximum(union, 1), np.nan)
    out: Dict[str, object] = {
        "n": len(pq.read_table(path, columns=["dish"])),
        "plate_group_mIoU": round(float(np.nanmean(iou)) * 100, 2),
        "plate_group_IoU": {EVAL_GROUPS[i]: round(float(iou[i]) * 100, 2) for i in range(5)},
        "ratio_211_MAE_pp": round(float(np.mean(maes)), 2),
        "ratio_211_MAE_median_pp": round(float(np.median(maes)), 2),
        "worst_dishes_211_MAE": dict(sorted(((k, round(float(np.mean(v)), 1)) for k, v in per_dish.items()), key=lambda x: -x[1])[:8]),
    }
    if use_cls:
        ci = _iou(cm_cls)
        out["class_mIoU"] = round(float(np.nanmean(ci)) * 100, 2)
        out["class_IoU"] = {CLASSES_V3[i]: (None if np.isnan(ci[i]) else round(float(ci[i]) * 100, 2)) for i in range(ncls)}
    return out


def eval_uec(m: Model, limit: int | None) -> Dict[str, object]:
    inter = union = 0
    n = 0
    for batch in pq.ParquetFile(os.path.join(CACHE, "uecfoodpix_val.parquet")).iter_batches(batch_size=32):
        for r in batch.to_pylist():
            if limit and n >= limit:
                break
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            gt = np.asarray(Image.open(io.BytesIO(r["label"]["bytes"])).convert("RGB"), dtype=np.uint8)[..., 0] > 0
            lab, _ = m.predict(im)
            gt = np.asarray(Image.fromarray(gt.astype(np.uint8)).resize(lab.shape[::-1], Image.NEAREST), dtype=bool)
            pr = lab > 0
            inter += int(np.count_nonzero(gt & pr))
            union += int(np.count_nonzero(gt | pr))
            n += 1
        if limit and n >= limit:
            break
    return {"n": n, "food_IoU": round(inter / max(union, 1) * 100, 2)}


def eval_nonfood(m: Model) -> Dict[str, object]:
    ids = set(json.load(open(os.path.join(CACHE, "tiny_nonfood_ids.json")))["non_food_ids"])
    rows = pq.ParquetFile(os.path.join(CACHE, "tiny_imagenet_val.parquet")).read_row_group(0).to_pylist()
    ok = n = 0
    for i, r in enumerate(rows):
        if i % 5 != 0 or int(r["label"]) not in ids:
            continue
        lab, fp = m.predict(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
        ok += int(not m.is_food(lab, fp))
        n += 1
        if n >= 300:
            break
    return {"n": n, "reject_pct": round(ok / max(n, 1) * 100, 1)}


def eval_thfood(m: Model) -> Dict[str, object]:
    d = os.path.join(CACHE, "thfood50_samples")
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    ok = 0
    for f in files:
        lab, fp = m.predict(Image.open(os.path.join(d, f)).convert("RGB"))
        ok += int(m.is_food(lab, fp))
    return {"n": len(files), "recognised_as_food_pct": round(ok / max(len(files), 1) * 100, 1)}


def main() -> None:
    onnx_path, meta_path = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    m = Model(onnx_path, meta_path)
    res = {"model": os.path.basename(onnx_path), "version": m.meta.get("version")}
    if os.environ.get("THAI_ONLY"):
        res["thai_test"] = eval_thai(m)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    res["thai_test"] = eval_thai(m)
    print(json.dumps(res["thai_test"], ensure_ascii=False), flush=True)
    res["foodseg103_val"] = eval_foodseg(m, limit)
    print(json.dumps(res["foodseg103_val"], ensure_ascii=False), flush=True)
    res["uecfoodpix_val"] = eval_uec(m, limit)
    res["nonfood_tiny_imagenet"] = eval_nonfood(m)
    res["thfood50_samples"] = eval_thfood(m)
    res["cpu_ms_median"] = round(float(np.median(m.ms)), 1)
    out = os.path.join(ROOT, "training", f"eval_{os.path.splitext(os.path.basename(onnx_path))[0]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
