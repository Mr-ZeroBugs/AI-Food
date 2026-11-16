"""
Compile a unified real-photo 5-Food-Group training & validation dataset into `datasets_cache/`:
1. FoodSeg103 train (4,983 human pixel-annotated food plates -> 10 nutrition classes)
2. Google Nutrition5k train (891 real overhead plates + exact weighed physical grams + teacher masks)
3. THFOOD-50 train (~2,500 balanced Thai meal photos across all 50 Thai menus via Tri-AI distillation)
4. Non-Food / Empty-Plate Negatives (800 real filtered Tiny-ImageNet non-food scenes)
"""

from __future__ import annotations

import json
import os
import random
import sys
from io import BytesIO
from typing import Dict, List

if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps
import torch

from food_taxonomy import FOODSEG103_ID2LABEL, FOODSEG103_TO_LOCAL_10CLASS
from nutrition_engine import CLASS_NAMES, FIVE_GROUP_KEYS, FOOD_GROUP_WEIGHTS, analyze_plate_from_mask
from real_foodseg103_engine import RealFoodSeg103Engine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "datasets_cache")
OUT_PARQUET = os.path.join(CACHE_DIR, "compiled_5group_train.parquet")

_FOODSEG_LOCAL_LUT = np.zeros(256, dtype=np.uint8)
for _cid, _lcid in FOODSEG103_TO_LOCAL_10CLASS.items():
    if 0 <= _cid < 256:
        _FOODSEG_LOCAL_LUT[_cid] = _lcid

_NAME_TO_FSID = {name.lower(): cid for cid, name in FOODSEG103_ID2LABEL.items()}


def _resize_max_side(im_pil: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = im_pil.size
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return im_pil
    return im_pil.resize(
        (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
        resample=Image.BILINEAR,
    )


def _encode_jpg_and_mask_png(img_rgb: np.ndarray, mask: np.ndarray, max_side: int = 512) -> tuple[bytes, bytes]:
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        im_pil = Image.fromarray(img_rgb).resize((nw, nh), resample=Image.BILINEAR)
        m_pil = Image.fromarray(mask).resize((nw, nh), resample=Image.NEAREST)
    else:
        im_pil = Image.fromarray(img_rgb)
        m_pil = Image.fromarray(mask)

    jbuf, pbuf = BytesIO(), BytesIO()
    im_pil.save(jbuf, format="JPEG", quality=90)
    m_pil.save(pbuf, format="PNG", compress_level=1)
    return jbuf.getvalue(), pbuf.getvalue()


def _ratios_vec(ratios_dict: Dict[str, float]) -> List[float]:
    return [round(float(ratios_dict.get(k, 0.0)) / 100.0, 5) for k in FIVE_GROUP_KEYS]


def compile_dataset(
    thfood_per_class: int = 50,
    nonfood_count: int = 800,
    seed: int = 42,
) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = RealFoodSeg103Engine(device=device)

    records: List[Dict[str, object]] = []

    # 1. FoodSeg103 train (4,983 human pixel-annotated images)
    print("\n[1/4] Compiling FoodSeg103 train (human ground-truth masks) ...", flush=True)
    for shard_idx in range(3):
        p = os.path.join(CACHE_DIR, f"foodseg103_train_{shard_idx}.parquet")
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                im = _resize_max_side(Image.open(BytesIO(row["image"]["bytes"])).convert("RGB"), 512)
                arr = np.asarray(im, dtype=np.uint8)
                lbl = np.asarray(Image.open(BytesIO(row["label"]["bytes"])), dtype=np.uint8)
                if lbl.ndim == 3:
                    lbl = lbl[..., 0]
                if lbl.shape != arr.shape[:2]:
                    lbl = np.asarray(
                        Image.fromarray(lbl).resize((arr.shape[1], arr.shape[0]), Image.NEAREST),
                        dtype=np.uint8,
                    )
                local_mask = _FOODSEG_LOCAL_LUT[lbl]
                an = analyze_plate_from_mask(local_mask, depth_map=None)
                if an["total_food_pixels"] == 0:
                    continue
                jb, pb = _encode_jpg_and_mask_png(arr, local_mask)
                records.append({
                    "image_bytes": jb,
                    "mask_bytes": pb,
                    "ratios": _ratios_vec(an["five_group_ratios_2d"]),
                    "is_food": 1.0,
                    "has_gram_gt": 0.0,
                    "source": "foodseg103",
                })
    print(f"  FoodSeg103 collected: {len(records)} plates", flush=True)

    # 2. Nutrition5k train (891 real overhead plates with weighed grams + teacher masks)
    print("\n[2/4] Compiling Google Nutrition5k train (weighed gram targets) ...", flush=True)
    n5k_count = 0
    pf_n5k = pq.ParquetFile(os.path.join(CACHE_DIR, "nutrition5k_train.parquet"))
    for batch in pf_n5k.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            im = _resize_max_side(Image.open(BytesIO(row["image"]["bytes"])).convert("RGB"), 512)
            arr = np.asarray(im, dtype=np.uint8)
            gt_g = {k: 0.0 for k in FIVE_GROUP_KEYS}
            for fs_n, meta in zip(row["fs103_classes"], row["ingredients"]):
                cid = _NAME_TO_FSID.get(str(fs_n).lower(), 0)
                lcid = FOODSEG103_TO_LOCAL_10CLASS.get(cid, 0)
                if lcid <= 0:
                    continue
                for gk, wt in FOOD_GROUP_WEIGHTS[CLASS_NAMES[lcid]].items():
                    gt_g[gk] += max(float(meta["grams"]), 0.0) * wt
            tot_g = sum(gt_g.values())
            if tot_g <= 0:
                continue
            gt_pct = {k: (v / tot_g) * 100.0 for k, v in gt_g.items()}
            pred_local = teacher.predict(arr)[0]
            if int(np.count_nonzero(pred_local > 0)) == 0:
                continue
            jb, pb = _encode_jpg_and_mask_png(arr, pred_local)
            records.append({
                "image_bytes": jb,
                "mask_bytes": pb,
                "ratios": _ratios_vec(gt_pct),
                "is_food": 1.0,
                "has_gram_gt": 1.0,
                "source": "nutrition5k",
            })
            n5k_count += 1
    print(f"  Nutrition5k collected: {n5k_count} weighed plates", flush=True)

    # 3. THFOOD-50 balanced Thai meals across all 50 Thai dish classes
    # First read ONLY labels (lightweight) to pick balanced row indices without loading 2.2 GB into RAM!
    print("\n[3/4] Distilling THFOOD-50 Thai dishes across all 50 classes ...", flush=True)
    shard_picks: Dict[int, Dict[int, int]] = {}  # shard_idx -> {row_idx: cls_id}
    per_cls_locs: Dict[int, List[tuple[int, int]]] = {}
    for s_idx in range(4):
        p = os.path.join(CACHE_DIR, f"thfood50_train_{s_idx}.parquet")
        if not os.path.exists(p):
            continue
        lbl_list = pq.read_table(p, columns=["label"])["label"].to_pylist()
        for r_idx, lbl in enumerate(lbl_list):
            per_cls_locs.setdefault(int(lbl), []).append((s_idx, r_idx))

    target_per_shard: Dict[int, set[int]] = {}
    for cls_id, locs in sorted(per_cls_locs.items()):
        rng.shuffle(locs)
        for s_idx, r_idx in locs[: int(thfood_per_class * 1.4)]:
            target_per_shard.setdefault(s_idx, set()).add(r_idx)
            shard_picks.setdefault(s_idx, {})[r_idx] = cls_id

    cls_done: Dict[int, int] = {}
    th_count = 0
    for s_idx in sorted(target_per_shard.keys()):
        p = os.path.join(CACHE_DIR, f"thfood50_train_{s_idx}.parquet")
        wanted = target_per_shard[s_idx]
        pf = pq.ParquetFile(p)
        cur_idx = 0
        for batch in pf.iter_batches(batch_size=32, columns=["image", "label"]):
            rows = batch.to_pylist()
            for r in rows:
                idx_now = cur_idx
                cur_idx += 1
                if idx_now not in wanted:
                    continue
                cls_id = int(r["label"])
                if cls_done.get(cls_id, 0) >= thfood_per_class:
                    continue
                try:
                    im_raw = ImageOps.exif_transpose(Image.open(BytesIO(r["image"]["bytes"]))).convert("RGB")
                    if min(im_raw.size) < 96:
                        continue
                    im = _resize_max_side(im_raw, max_side=512)
                    arr = np.asarray(im, dtype=np.uint8)
                    pred_local, _raw, conf, _lat, _det, _dish = teacher.predict(arr)
                    if conf < 0.50 or int(np.count_nonzero(pred_local > 0)) < 150:
                        continue
                    an = analyze_plate_from_mask(pred_local, depth_map=None)
                    jb, pb = _encode_jpg_and_mask_png(arr, pred_local)
                    records.append({
                        "image_bytes": jb,
                        "mask_bytes": pb,
                        "ratios": _ratios_vec(an["five_group_ratios_2d"]),
                        "is_food": 1.0,
                        "has_gram_gt": 0.0,
                        "source": "thfood50",
                    })
                    cls_done[cls_id] = cls_done.get(cls_id, 0) + 1
                    th_count += 1
                    if th_count % 250 == 0:
                        print(f"    THFOOD-50 progress: {th_count} Thai plates ...", flush=True)
                except Exception:
                    continue
    print(f"  THFOOD-50 distilled: {th_count} real Thai dishes ({len(cls_done)} classes)", flush=True)

    # 4. Strictly Non-Food Real Negatives (Tiny-ImageNet filtered holdout-free 80% train slice)
    print("\n[4/4] Adding strictly Non-Food negative scenes ...", flush=True)
    nonfood_ids = set(json.load(open(os.path.join(CACHE_DIR, "tiny_nonfood_ids.json"), "r"))["non_food_ids"])
    nf_pool: List[bytes] = []
    pf_tiny = pq.ParquetFile(os.path.join(CACHE_DIR, "tiny_imagenet_val.parquet"))
    row_counter = 0
    for batch in pf_tiny.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if int(row["label"]) in nonfood_ids and (row_counter % 5 != 0):
                nf_pool.append(row["image"]["bytes"])
            row_counter += 1
    rng.shuffle(nf_pool)
    nf_count = 0
    for img_b in nf_pool[:nonfood_count]:
        im = Image.open(BytesIO(img_b)).convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        zero_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        jb, pb = _encode_jpg_and_mask_png(arr, zero_mask)
        records.append({
            "image_bytes": jb,
            "mask_bytes": pb,
            "ratios": [0.0, 0.0, 0.0, 0.0, 0.0],
            "is_food": 0.0,
            "has_gram_gt": 0.0,
            "source": "nonfood",
        })
        nf_count += 1
    print(f"  Non-Food negatives collected: {nf_count}", flush=True)

    rng.shuffle(records)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, OUT_PARQUET, row_group_size=256)
    print(f"\nDONE! Compiled {len(records)} real samples -> {OUT_PARQUET} ({os.path.getsize(OUT_PARQUET)/1024/1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    compile_dataset()
