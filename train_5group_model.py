"""
Train a Multi-Task SegFormer-B2 directly for:
  1. Direct 10-Class -> 5-Food-Group Pixel Segmentation
  2. Out-of-Distribution Food-vs-NonFood Gate (`is_food`)
  3. 5-Food-Group Global Soft Portion/Gram Regression Head

Exports best checkpoint to `models/segformer_b2_5group_multitask.pt` and
quantized INT8 ONNX to `models_onnx/segformer_5group_int8.onnx.gz`.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import time
from io import BytesIO
from typing import Dict, List, Tuple

if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSemanticSegmentation

from food_taxonomy import FOODSEG103_TO_LOCAL_10CLASS
from nutrition_engine import FIVE_GROUP_KEYS, analyze_plate_from_mask


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "datasets_cache")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")
TRAIN_PARQUET = os.path.join(CACHE_DIR, "compiled_5group_train.parquet")
VAL_FOODSEG_PARQUET = os.path.join(CACHE_DIR, "foodseg103_val.parquet")
VAL_TINY_PARQUET = os.path.join(CACHE_DIR, "tiny_imagenet_val.parquet")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FS_LUT = np.zeros(256, dtype=np.uint8)
for _cid, _lcid in FOODSEG103_TO_LOCAL_10CLASS.items():
    if 0 <= _cid < 256:
        _FS_LUT[_cid] = _lcid


def _prepare_edgepad_512(
    im_pil: Image.Image,
    mask_np: np.ndarray,
    is_train: bool = True,
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving resize + EdgePad to (512, 512) RGB & (128, 128) label mask."""
    r = rng or random
    w, h = im_pil.size

    if is_train:
        if r.random() < 0.5:
            im_pil = im_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_np = np.fliplr(mask_np).copy()
        if r.random() < 0.15:
            im_pil = im_pil.transpose(Image.FLIP_TOP_BOTTOM)
            mask_np = np.flipud(mask_np).copy()
        if r.random() < 0.65:
            im_pil = ImageEnhance.Brightness(im_pil).enhance(r.uniform(0.88, 1.12))
            im_pil = ImageEnhance.Contrast(im_pil).enhance(r.uniform(0.88, 1.12))
            im_pil = ImageEnhance.Color(im_pil).enhance(r.uniform(0.90, 1.10))
        target_max = r.randint(448, 512)
    else:
        target_max = 512

    scale = float(target_max) / max(w, h)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    im_rs = np.asarray(im_pil.resize((nw, nh), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    m_rs = np.asarray(
        Image.fromarray(mask_np).resize((nw, nh), resample=Image.NEAREST), dtype=np.uint8
    )

    pv = ((im_rs - _MEAN) / _STD).transpose(2, 0, 1)
    if nh < 512 or nw < 512:
        pv_512 = np.pad(pv, ((0, 0), (0, 512 - nh), (0, 512 - nw)), mode="edge")
        m_512 = np.pad(m_rs, ((0, 512 - nh), (0, 512 - nw)), mode="constant", constant_values=0)
    else:
        pv_512 = pv[:, :512, :512]
        m_512 = m_rs[:512, :512]

    m_128 = np.asarray(
        Image.fromarray(m_512).resize((128, 128), resample=Image.NEAREST), dtype=np.int64
    )
    return pv_512.astype(np.float32), m_128


class Compiled5GroupDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], is_train: bool = True) -> None:
        self.rows = rows
        self.is_train = is_train
        self.rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        im = Image.open(BytesIO(row["image_bytes"])).convert("RGB")  # type: ignore[arg-type]
        m = np.asarray(Image.open(BytesIO(row["mask_bytes"])), dtype=np.uint8)  # type: ignore[arg-type]
        pv, m128 = _prepare_edgepad_512(im, m, is_train=self.is_train, rng=self.rng)
        ratios = np.asarray(row["ratios"], dtype=np.float32)  # type: ignore[arg-type]
        is_food = np.float32(row["is_food"])  # type: ignore[arg-type]
        has_gram = np.float32(row["has_gram_gt"])  # type: ignore[arg-type]
        return (
            torch.from_numpy(pv),
            torch.from_numpy(m128),
            torch.from_numpy(ratios),
            torch.tensor(is_food, dtype=torch.float32),
            torch.tensor(has_gram, dtype=torch.float32),
        )


class MultiTaskSegFormer5Group(nn.Module):
    """Unified SegFormer-B2 with 10-class Segmentation + Food OOD Gate + 5-Group Ratio Head."""

    def __init__(self, backbone_repo: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> None:
        super().__init__()
        self.segformer = AutoModelForSemanticSegmentation.from_pretrained(
            backbone_repo,
            num_labels=10,
            ignore_mismatched_sizes=True,
        )
        hidden_dim = int(self.segformer.config.hidden_sizes[-1])  # 512 for MiT-B2
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.food_head = nn.Linear(256, 1)
        self.ratio_head = nn.Linear(256, 5)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.segformer(pixel_values=pixel_values, output_hidden_states=True)
        seg_logits = out.logits  # (B, 10, 128, 128)
        feat = out.hidden_states[-1]  # (B, 512, 16, 16)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        emb = self.shared_proj(pooled)
        food_logit = self.food_head(emb)  # (B, 1)
        group_ratios = torch.softmax(self.ratio_head(emb), dim=-1)  # (B, 5)
        return seg_logits, food_logit, group_ratios


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3)) + 1e-5
    dice_per_cls = (2.0 * inter + 1e-5) / denom
    # Only average over classes present in batch (excluding background 0 dominance)
    present = onehot.sum(dim=(0, 2, 3)) > 0
    if torch.any(present):
        return 1.0 - dice_per_cls[present].mean()
    return 1.0 - dice_per_cls.mean()


@torch.no_grad()
def evaluate_validation(
    model: MultiTaskSegFormer5Group,
    device: torch.device,
    val_limit: int = 120,
) -> Dict[str, float]:
    """Evaluate strictly on held-out FoodSeg103 Validation + Tiny-ImageNet Non-Food."""
    model.eval()
    pf = pq.ParquetFile(VAL_FOODSEG_PARQUET)
    rows = next(pf.iter_batches(batch_size=val_limit)).to_pylist()
    eval5_lut = np.array([0, 2, 2, 1, 1, 1, 3, 3, 4, 2], dtype=np.int64)
    cm = np.zeros((5, 5), dtype=np.int64)
    maes: List[float] = []
    food_tp = 0

    for row in rows:
        im = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        gt_raw = np.asarray(Image.open(BytesIO(row["label"]["bytes"])), dtype=np.uint8)
        if gt_raw.ndim == 3:
            gt_raw = gt_raw[..., 0]
        w, h = im.size
        if gt_raw.shape != (h, w):
            gt_raw = np.asarray(Image.fromarray(gt_raw).resize((w, h), Image.NEAREST), dtype=np.uint8)
        gt_local = _FS_LUT[gt_raw]

        scale = 512.0 / max(w, h)
        nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        im_rs = np.asarray(im.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = ((im_rs - _MEAN) / _STD).transpose(2, 0, 1)
        pv_512 = np.pad(pv, ((0, 0), (0, 512 - nh), (0, 512 - nw)), mode="edge")[None, ...]

        seg_logits, food_logit, _ratios = model(torch.from_numpy(pv_512).to(device))
        if torch.sigmoid(food_logit[0, 0]).item() >= 0.5:
            food_tp += 1

        ch, cw = max(1, int(round(nh / 4.0))), max(1, int(round(nw / 4.0)))
        pred_128 = seg_logits[0, :, :ch, :cw].argmax(dim=0).cpu().numpy().astype(np.uint8)
        pred_local = np.asarray(Image.fromarray(pred_128).resize((w, h), Image.NEAREST), dtype=np.uint8)

        gt_e = eval5_lut[gt_local].ravel()
        pr_e = eval5_lut[pred_local].ravel()
        cm += np.bincount(gt_e * 5 + pr_e, minlength=25).reshape(5, 5)

        gt_r = analyze_plate_from_mask(gt_local)["five_group_ratios_2d"]
        pr_r = analyze_plate_from_mask(pred_local)["five_group_ratios_2d"]
        maes.append(float(np.mean([abs(pr_r[k] - gt_r[k]) for k in FIVE_GROUP_KEYS])))

    diag = np.diag(cm).astype(np.float64)
    union = (cm.sum(axis=1) + cm.sum(axis=0) - diag).astype(np.float64)
    valid = union > 0
    miou = float(np.mean(diag[valid] / union[valid]) * 100.0)
    pix_acc = float(diag.sum() / max(cm.sum(), 1) * 100.0)

    # Check non-food rejection strictly on held-out 20% deterministic split (idx % 5 == 0)
    nonfood_ids = set(json.load(open(os.path.join(CACHE_DIR, "tiny_nonfood_ids.json"), "r"))["non_food_ids"])
    pf_t = pq.ParquetFile(VAL_TINY_PARQUET)
    t_rows = pf_t.read_row_group(0).to_pylist()
    nf_correct, nf_total = 0, 0
    for r_idx, row in enumerate(t_rows):
        if r_idx % 5 != 0 or int(row["label"]) not in nonfood_ids:
            continue
        im = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB").resize((512, 512), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        pv = ((arr - _MEAN) / _STD).transpose(2, 0, 1)[None, ...]
        _sl, food_logit, _ = model(torch.from_numpy(pv).to(device))
        if torch.sigmoid(food_logit[0, 0]).item() < 0.5:
            nf_correct += 1
        nf_total += 1
        if nf_total >= 100:
            break

    return {
        "mIoU_pct": round(miou, 2),
        "pixel_acc_pct": round(pix_acc, 2),
        "MAE_pct_points": round(float(np.mean(maes)), 2),
        "food_recall_pct": round((food_tp / max(len(rows), 1)) * 100.0, 1),
        "nonfood_reject_pct": round((nf_correct / max(nf_total, 1)) * 100.0, 1),
    }


def export_onnx_and_quantize(ckpt_path: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = MultiTaskSegFormer5Group()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(ONNX_DIR, exist_ok=True)
    fp32_path = os.path.join(MODELS_DIR, "segformer_5group_fp32.onnx")
    int8_path = os.path.join(MODELS_DIR, "segformer_5group_int8.onnx")
    gz_path = os.path.join(ONNX_DIR, "segformer_5group_int8.onnx.gz")

    dummy = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model,
        (dummy,),
        fp32_path,
        input_names=["pixel_values"],
        output_names=["seg_logits", "food_logit", "group_ratios"],
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
    )
    with open(int8_path, "rb") as fin, gzip.open(gz_path, "wb", compresslevel=6) as fout:
        fout.write(fin.read())
    print(
        f"Exported ONNX INT8: raw={os.path.getsize(int8_path)/1024/1024:.2f} MB | "
        f"gz={os.path.getsize(gz_path)/1024/1024:.2f} MB -> {gz_path}",
        flush=True,
    )
    return gz_path


def train(epochs: int = 6, batch_size: int = 4, grad_accum: int = 3) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Loading compiled training parquet: {TRAIN_PARQUET}", flush=True)
    table = pq.read_table(TRAIN_PARQUET)
    rows = table.to_pylist()
    print(f"Total compiled training samples: {len(rows)}", flush=True)

    ds = Compiled5GroupDataset(rows, is_train=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiTaskSegFormer5Group().to(device)
    backbone_params = list(model.segformer.segformer.parameters())
    head_params = (
        list(model.segformer.decode_head.parameters())
        + list(model.shared_proj.parameters())
        + list(model.food_head.parameters())
        + list(model.ratio_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 8e-5},
            {"params": head_params, "lr": 4e-4},
        ],
        weight_decay=0.01,
    )
    opt_steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = epochs * opt_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    cls_weights = torch.tensor(
        [0.45, 1.0, 1.35, 0.95, 1.8, 1.8, 1.0, 1.1, 1.6, 1.35],
        dtype=torch.float32,
        device=device,
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.03)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_score = -1e9
    best_ckpt = os.path.join(MODELS_DIR, "segformer_b2_5group_multitask.pt")

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (pv, m128, ratios, is_food, has_gram) in enumerate(loader, 1):
            pv = pv.to(device, non_blocking=True)
            m128 = m128.to(device, non_blocking=True)
            ratios = ratios.to(device, non_blocking=True)
            is_food = is_food.to(device, non_blocking=True)
            has_gram = has_gram.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                seg_logits, food_logit, pred_ratios = model(pv)
                loss_ce = ce_loss_fn(seg_logits, m128)
                loss_dice = _soft_dice_loss(seg_logits, m128, num_classes=10)
                loss_ood = bce_loss_fn(food_logit.squeeze(1), is_food)

                # Supervise global 5-group ratios strictly on food samples (2x weight for real N5K grams)
                food_mask = is_food > 0.5
                if torch.any(food_mask):
                    sample_wt = torch.where(has_gram[food_mask] > 0.5, 2.0, 1.0)
                    l1 = torch.abs(pred_ratios[food_mask] - ratios[food_mask]).mean(dim=1)
                    loss_ratio = (l1 * sample_wt).mean()
                else:
                    loss_ratio = torch.tensor(0.0, device=device)

                raw_loss = loss_ce + 0.65 * loss_dice + 0.35 * loss_ood + 1.5 * loss_ratio
                loss = raw_loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.item())

            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 150 == 0 or step == len(loader):
                dt = time.perf_counter() - t0
                print(
                    f"  Epoch {ep}/{epochs} [{step}/{len(loader)}] loss={running_loss/step:.4f} ({dt:.1f}s)",
                    flush=True,
                )

        val_metrics = evaluate_validation(model, device=device, val_limit=120)
        elapsed = time.perf_counter() - t0
        composite = val_metrics["mIoU_pct"] - 1.5 * val_metrics["MAE_pct_points"] + 0.2 * val_metrics["nonfood_reject_pct"]
        print(
            f"[Epoch {ep}/{epochs} | {elapsed:.1f}s] Val mIoU={val_metrics['mIoU_pct']}% | "
            f"PixAcc={val_metrics['pixel_acc_pct']}% | MAE={val_metrics['MAE_pct_points']}% | "
            f"FoodRecall={val_metrics['food_recall_pct']}% | NonFoodReject={val_metrics['nonfood_reject_pct']}%",
            flush=True,
        )
        if composite > best_score:
            best_score = composite
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Saved new best checkpoint ({best_ckpt})", flush=True)

    export_onnx_and_quantize(best_ckpt)


if __name__ == "__main__":
    train()
