"""Train the v3 SegFormer-B2 food segmenter (11 classes aligned to the Thai 5 food groups).

Two heads in one pass:
  1. seg_logits  (B, 11, 128, 128)  per-pixel v3 class (see backend/taxonomy.py)
  2. food_logit  (B, 1)             image-level food vs non-food gate

The old "global ratio" head is gone: plate proportions now come only from the mask, so the
numbers on screen always match the painted regions.

Usage (from repo root):
    python training/train_segformer_v3.py build      # compile datasets_cache/compiled_v3_train.parquet
    python training/train_segformer_v3.py train      # train + export ONNX to static/models/
    python training/train_segformer_v3.py export     # export the saved checkpoint only
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]  # broken torchvision wheel on Windows

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.taxonomy import CLASSES_V3, FOODSEG103_TO_V3, V2_TO_V3  # noqa: E402

CACHE = os.path.join(ROOT, "datasets_cache")
MODELS = os.path.join(ROOT, "models")
WEB_MODELS = os.path.join(ROOT, "public", "models")
COMPILED_V2 = os.path.join(CACHE, "compiled_5group_train.parquet")
COMPILED_V3 = os.path.join(CACHE, "compiled_v3_train.parquet")
VAL_FS = os.path.join(CACHE, "foodseg103_val.parquet")
TINY = os.path.join(CACHE, "tiny_imagenet_val.parquet")
# RUN_TAG names the run's checkpoint/exports; RUN_DATA picks the training parquet.
TAG = os.environ.get("RUN_TAG", "v3")
TRAIN_DATA = os.environ.get("RUN_DATA", COMPILED_V3)
CKPT = os.path.join(MODELS, f"segformer_b2_{TAG}.pt")
NUM_CLASSES = len(CLASSES_V3)
INPUT = 512

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FS_LUT = np.zeros(256, dtype=np.uint8)
for _k, _v in FOODSEG103_TO_V3.items():
    FS_LUT[_k] = _v
V2_LUT = np.zeros(256, dtype=np.uint8)
V2_LUT[: len(V2_TO_V3)] = V2_TO_V3

# Group index used for group-level metrics: 0 bg, 1 protein, 2 carb, 3 veg, 4 fruit, 5 other(mixed/sweets/liquid)
CLASS_TO_EVAL_GROUP = np.array([0, 2, 2, 1, 1, 1, 3, 4, 5, 5, 5], dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset compilation
# ---------------------------------------------------------------------------

def _resize_max(im: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s >= 1.0:
        return im
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def _encode(img: Image.Image, mask: np.ndarray) -> Tuple[bytes, bytes]:
    jb = io.BytesIO()
    img.save(jb, format="JPEG", quality=92)
    pb = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(pb, format="PNG")
    return jb.getvalue(), pb.getvalue()


def build() -> None:
    rows: List[Dict[str, object]] = []
    # 1. FoodSeg103 train: human masks, remapped from the raw 103 ids.
    for shard in range(3):
        pf = pq.ParquetFile(os.path.join(CACHE, f"foodseg103_train_{shard}.parquet"))
        for batch in pf.iter_batches(batch_size=128, columns=["image", "label"]):
            for r in batch.to_pylist():
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                lbl = np.asarray(Image.open(io.BytesIO(r["label"]["bytes"])), dtype=np.uint8)
                if lbl.ndim == 3:
                    lbl = lbl[..., 0]
                im = _resize_max(im)
                lbl = np.asarray(Image.fromarray(lbl).resize(im.size, Image.NEAREST), dtype=np.uint8)
                m = FS_LUT[lbl]
                if not np.any(m > 0):
                    continue
                jb, pb = _encode(im, m)
                rows.append({"image_bytes": jb, "mask_bytes": pb, "is_food": 1.0, "source": "foodseg103"})
        print(f"  FoodSeg103 shard {shard}: total rows {len(rows)}", flush=True)

    # 2. THFOOD-50 / Nutrition5k pseudo-labelled rows + Tiny-ImageNet non-food, remapped v2 -> v3.
    pf = pq.ParquetFile(COMPILED_V2)
    kept = {"thfood50": 0, "nutrition5k": 0, "nonfood": 0}
    for batch in pf.iter_batches(batch_size=256, columns=["image_bytes", "mask_bytes", "is_food", "source"]):
        for r in batch.to_pylist():
            if r["source"] not in kept:
                continue
            m = np.asarray(Image.open(io.BytesIO(r["mask_bytes"])), dtype=np.uint8)
            m3 = V2_LUT[m]
            pb = io.BytesIO()
            Image.fromarray(m3).save(pb, format="PNG")
            rows.append({"image_bytes": r["image_bytes"], "mask_bytes": pb.getvalue(),
                         "is_food": float(r["is_food"]), "source": r["source"]})
            kept[r["source"]] += 1
    print(f"  Reused from v2 compile: {kept}", flush=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, COMPILED_V3)
    print(f"Wrote {len(rows)} rows -> {COMPILED_V3}", flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForSemanticSegmentation
    return torch, nn, F, AutoModelForSemanticSegmentation


def build_model():
    torch, nn, _F, AutoSeg = _torch()

    class FoodSegV3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.segformer = AutoSeg.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
            hid = int(self.segformer.config.hidden_sizes[-1])
            self.food_head = nn.Sequential(
                nn.Linear(hid * 2, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 1)
            )

        def forward(self, pixel_values):
            out = self.segformer(pixel_values=pixel_values, output_hidden_states=True)
            feat = out.hidden_states[-1]
            pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
            return out.logits, self.food_head(pooled)

    return FoodSegV3()


def preprocess(im: Image.Image, mask: np.ndarray | None, train: bool, rng: random.Random):
    """Aspect-preserving resize, edge-pad bottom/right to 512. Must match static/js/segmenter.js."""
    if train:
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy() if mask is not None else None
        if rng.random() < 0.7:
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
            im = ImageEnhance.Color(im).enhance(rng.uniform(0.8, 1.2))
        target = rng.randint(384, INPUT)
    else:
        target = INPUT
    w, h = im.size
    s = target / max(w, h)
    nw, nh = max(16, round(w * s)), max(16, round(h * s))
    arr = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    pv = ((arr - MEAN) / STD).transpose(2, 0, 1)
    pv = np.pad(pv, ((0, 0), (0, INPUT - nh), (0, INPUT - nw)), mode="edge")
    m128 = None
    if mask is not None:
        m = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.NEAREST), dtype=np.uint8)
        m = np.pad(m, ((0, INPUT - nh), (0, INPUT - nw)), mode="constant", constant_values=255)
        m128 = np.asarray(Image.fromarray(m).resize((128, 128), Image.NEAREST), dtype=np.int64)
    return pv.astype(np.float32), m128, (nh, nw)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, device, limit: int | None = 500) -> Dict[str, float]:
    torch, *_ = _torch()
    model.eval()
    cm_cls = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_grp = np.zeros((6, 6), dtype=np.int64)
    n = 0
    food_ok = 0
    with torch.no_grad():
        pf = pq.ParquetFile(VAL_FS)
        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            for r in batch.to_pylist():
                if limit is not None and n >= limit:
                    break
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                lbl = np.asarray(Image.open(io.BytesIO(r["label"]["bytes"])), dtype=np.uint8)
                if lbl.ndim == 3:
                    lbl = lbl[..., 0]
                gt = FS_LUT[lbl]
                pv, _, (nh, nw) = preprocess(im, None, False, random.Random(0))
                seg, food = model(torch.from_numpy(pv[None]).to(device))
                food_ok += int(torch.sigmoid(food[0, 0]).item() >= 0.5)
                ch, cw = max(1, round(nh / 4)), max(1, round(nw / 4))
                pred = seg[0, :, :ch, :cw].argmax(0).cpu().numpy().astype(np.uint8)
                pred = np.asarray(Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST), dtype=np.int64)
                g = gt.astype(np.int64).ravel()
                p = pred.ravel()
                cm_cls += np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                ge, pe = CLASS_TO_EVAL_GROUP[g], CLASS_TO_EVAL_GROUP[p]
                cm_grp += np.bincount(ge * 6 + pe, minlength=36).reshape(6, 6)
                n += 1
            if limit is not None and n >= limit:
                break

        # Non-food gate on the held-out Tiny-ImageNet split (idx % 5 == 0, never trained on).
        ids = set(json.load(open(os.path.join(CACHE, "tiny_nonfood_ids.json")))["non_food_ids"])
        t_rows = pq.ParquetFile(TINY).read_row_group(0).to_pylist()
        nf_ok = nf_n = 0
        for i, r in enumerate(t_rows):
            if i % 5 != 0 or int(r["label"]) not in ids:
                continue
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            pv, _, _ = preprocess(im, None, False, random.Random(0))
            _, food = model(torch.from_numpy(pv[None]).to(device))
            nf_ok += int(torch.sigmoid(food[0, 0]).item() < 0.5)
            nf_n += 1
            if nf_n >= 300:
                break

    def miou(cm):
        d = np.diag(cm).astype(float)
        u = cm.sum(0) + cm.sum(1) - d
        v = u > 0
        return float(np.mean(d[v] / u[v]) * 100), {i: round(float(d[i] / u[i] * 100), 2) for i in range(len(d)) if u[i] > 0}

    cls_miou, cls_iou = miou(cm_cls)
    grp_miou, grp_iou = miou(cm_grp)
    return {
        "n_val": n,
        "class_mIoU": round(cls_miou, 2),
        "class_IoU": {CLASSES_V3[k]: v for k, v in cls_iou.items()},
        "group_mIoU": round(grp_miou, 2),
        "group_IoU": {["background", "protein", "carb", "veg", "fruit", "other"][k]: v for k, v in grp_iou.items()},
        "pixel_acc": round(float(np.trace(cm_cls) / max(cm_cls.sum(), 1) * 100), 2),
        "food_recall": round(food_ok / max(n, 1) * 100, 1),
        "nonfood_reject_tiny_imagenet": round(nf_ok / max(nf_n, 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int = 8, batch_size: int = 4, grad_accum: int = 3) -> None:
    torch, nn, F, _ = _torch()
    from torch.utils.data import DataLoader, Dataset

    os.makedirs(MODELS, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pq.read_table(TRAIN_DATA).to_pylist()
    print(f"Training rows: {len(rows)}", flush=True)

    class DS(Dataset):
        def __init__(self) -> None:
            self.rng = random.Random(42)

        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            im = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
            m = np.asarray(Image.open(io.BytesIO(r["mask_bytes"])), dtype=np.uint8)
            pv, m128, _ = preprocess(im, m, True, self.rng)
            return torch.from_numpy(pv), torch.from_numpy(m128), torch.tensor(float(r["is_food"]))

    loader = DataLoader(DS(), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                        pin_memory=device.type == "cuda")
    model = build_model().to(device)
    backbone = list(model.segformer.segformer.parameters())
    heads = list(model.segformer.decode_head.parameters()) + list(model.food_head.parameters())
    opt = torch.optim.AdamW([{"params": backbone, "lr": 6e-5}, {"params": heads, "lr": 4e-4}], weight_decay=0.01)
    total = epochs * math.ceil(len(loader) / grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[6e-5, 4e-4], total_steps=total, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # bg, rice, starch, animal, egg, plant, veg, fruit, mixed, sweets, liquid
    weights = torch.tensor([0.5, 1.0, 1.1, 1.0, 1.4, 1.5, 1.0, 1.4, 1.2, 1.3, 1.2], device=device)
    ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255, label_smoothing=0.03)
    bce = nn.BCEWithLogitsLoss()

    def dice(logits, target):
        valid = (target != 255).unsqueeze(1).float()
        probs = torch.softmax(logits, 1) * valid
        oh = F.one_hot(target.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float() * valid
        inter = (probs * oh).sum((0, 2, 3))
        den = probs.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1e-5
        present = oh.sum((0, 2, 3)) > 0
        d = (2 * inter + 1e-5) / den
        return 1 - (d[present].mean() if present.any() else d.mean())

    best = -1.0
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (pv, m, food) in enumerate(loader, 1):
            pv, m, food = pv.to(device), m.to(device), food.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                seg, fl = model(pv)
                # Non-food images carry an all-background mask, which is correct supervision.
                loss = ce(seg, m) + 0.6 * dice(seg, m) + 0.4 * bce(fl.squeeze(1), food)
            scaler.scale(loss / grad_accum).backward()
            run += float(loss.item())
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if sched.last_epoch + 1 < total:
                    sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"  ep {ep} [{step}/{len(loader)}] loss {run / step:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        metrics = evaluate(model, device, limit=500)
        metrics["epoch"] = ep
        history.append(metrics)
        print(f"[epoch {ep}] {json.dumps(metrics, ensure_ascii=False)}", flush=True)
        score = metrics["group_mIoU"]
        if score > best:
            best = score
            torch.save(model.state_dict(), CKPT)
            print(f"  saved best -> {CKPT}", flush=True)
        with open(os.path.join(MODELS, f"segformer_b2_{TAG}_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    export()


def export() -> None:
    torch, *_ = _torch()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = build_model()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()
    os.makedirs(WEB_MODELS, exist_ok=True)
    fp32 = os.path.join(MODELS, f"segformer_{TAG}_fp32.onnx")
    int8 = os.path.join(WEB_MODELS, f"segformer_{TAG}_int8.onnx")
    torch.onnx.export(model, (torch.randn(1, 3, INPUT, INPUT),), fp32, input_names=["pixel_values"],
                      output_names=["seg_logits", "food_logit"], opset_version=17, dynamo=False)
    quantize_dynamic(fp32, int8, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QUInt8)
    meta = {"file": f"segformer_{TAG}_int8.onnx", "version": TAG, "classes": CLASSES_V3, "input": INPUT,
            "output_stride": 4, "mean": MEAN.tolist(), "std": STD.tolist()}
    with open(os.path.join(WEB_MODELS, f"segformer_{TAG}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {int8} ({os.path.getsize(int8) / 1e6:.1f} MB)", flush=True)

    # Full-validation numbers for the README (all 2,135 FoodSeg103 val images).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = evaluate(model.to(device), device, limit=None)
    with open(os.path.join(ROOT, "training", f"eval_{TAG}_foodseg103_full.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    print(f"Full val: {json.dumps(full, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"build": build, "train": train, "export": export}[cmd]()
