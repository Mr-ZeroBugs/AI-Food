"""
Surgical 2-epoch refinement of `segformer_b2_5group_multitask.pt` using real THFOOD-50
Thai Egg dishes (`KaiJeowMooSaap`, `KaiThoon`, `SonInLawEggs`) + replay from
`compiled_5group_train.parquet` so crispy golden omelettes map cleanly to `egg_fried`
instead of `fried_rice`, with zero catastrophic forgetting on FoodSeg103 / Non-Food.
"""

from __future__ import annotations

import glob
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
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nutrition_engine import FIVE_GROUP_KEYS, analyze_plate_from_mask
from real_foodseg103_engine import RealFoodSeg103Engine
from train_5group_model import (
    BASE_DIR,
    CACHE_DIR,
    MODELS_DIR,
    TRAIN_PARQUET,
    Compiled5GroupDataset,
    MultiTaskSegFormer5Group,
    _MEAN,
    _STD,
    _soft_dice_loss,
    evaluate_validation,
    export_onnx_and_quantize,
)


EGG_LABEL_IDS = {17, 18, 43}  # KaiJeowMooSaap, KaiThoon, SonInLawEggs


def _encode_png(arr: np.ndarray, mode: str) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _encode_jpeg(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _build_egg_and_replay_rows(
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = RealFoodSeg103Engine(device=device)

    train_rows: List[Dict[str, object]] = []
    val_egg_imgs: List[np.ndarray] = []

    print("[1/3] Generating clean masks for real THFOOD-50 egg dishes ...", flush=True)
    egg_idx = 0
    for fpath in sorted(glob.glob(os.path.join(CACHE_DIR, "thfood50_train_*.parquet"))):
        tbl = pq.read_table(fpath, columns=["image", "label"])
        for im_obj, lbl in zip(tbl["image"].to_pylist(), tbl["label"].to_pylist()):
            if int(lbl) not in EGG_LABEL_IDS:
                continue
            im = Image.open(BytesIO(im_obj["bytes"])).convert("RGB")
            im.thumbnail((512, 512), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            if egg_idx % 7 == 0:
                val_egg_imgs.append(arr)
                egg_idx += 1
                continue
            egg_idx += 1

            lmask, _raw, _conf, _lat, _ings, _th = teacher.predict(arr)
            # Ensure remaining golden non-rice carb leaks inside verified KaiJeow/KaiThoon map cleanly to egg_fried (4)
            if int(lbl) in (17, 18):
                lmask = lmask.copy()
                lmask[(lmask == 9) | (lmask == 2)] = 4
            if int(np.count_nonzero((lmask == 4) | (lmask == 5))) < 100:
                continue

            an = analyze_plate_from_mask(lmask)
            r_vec = [float(an["five_group_ratios_2d"][k]) / 100.0 for k in FIVE_GROUP_KEYS]
            train_rows.append(
                {
                    "source": "thfood50_egg_clean",
                    "image_bytes": _encode_jpeg(arr),
                    "mask_bytes": _encode_png(lmask.astype(np.uint8), "L"),
                    "ratios": r_vec,
                    "is_food": 1.0,
                    "has_gram_gt": 0.0,
                }
            )
    n_egg = len(train_rows)
    print(f"  Prepared {n_egg} clean Thai egg train plates + {len(val_egg_imgs)} unseen holdout plates", flush=True)

    # Free teacher VRAM cleanly before SegFormer fine-tuning
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[2/3] Sampling stratified replay from compiled_5group_train.parquet ...", flush=True)
    tbl = pq.read_table(TRAIN_PARQUET)
    all_rows = tbl.to_pylist()
    rng = random.Random(seed)
    by_src: Dict[str, List[Dict[str, object]]] = {}
    for row in all_rows:
        src = str(row["source"])
        by_src.setdefault(src, []).append(row)

    quotas = {
        "foodseg103": 750,
        "nutrition5k": 550,
        "thfood50": 450,
        "tiny_nonfood": 250,
    }
    for src, cap in quotas.items():
        pool = list(by_src.get(src, []))
        rng.shuffle(pool)
        train_rows.extend(pool[:cap])

    rng.shuffle(train_rows)
    print(f"  Total refinement batch size: {len(train_rows)} ({n_egg} clean egg + replay)", flush=True)
    return train_rows, val_egg_imgs


@torch.no_grad()
def _eval_egg_holdout(model: MultiTaskSegFormer5Group, imgs: List[np.ndarray], device: torch.device) -> Tuple[float, float]:
    model.eval()
    egg_hits = 0
    fried_rice_leaks = 0
    for arr in imgs:
        h, w = arr.shape[:2]
        scale = 512.0 / max(w, h)
        nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        im_rs = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        pv = np.pad(((im_rs - _MEAN) / _STD).transpose(2, 0, 1), ((0, 0), (0, 512 - nh), (0, 512 - nw)), mode="edge")[None, ...]
        seg_l, _, _ = model(torch.from_numpy(pv).to(device))
        ch, cw = max(1, int(round(nh / 4.0))), max(1, int(round(nw / 4.0)))
        pred = seg_l[0, :, :ch, :cw].argmax(dim=0).cpu().numpy()
        egg_px = int(np.count_nonzero((pred == 4) | (pred == 5)))
        fr_px = int(np.count_nonzero(pred == 9))
        if egg_px > fr_px and egg_px > 20:
            egg_hits += 1
        if fr_px > egg_px:
            fried_rice_leaks += 1
    n = max(len(imgs), 1)
    return round(egg_hits / n * 100.0, 1), round(fried_rice_leaks / n * 100.0, 1)


def main() -> None:
    ckpt_path = os.path.join(MODELS_DIR, "segformer_b2_5group_multitask.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_rows, val_egg_imgs = _build_egg_and_replay_rows()

    model = MultiTaskSegFormer5Group()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    before_fs = evaluate_validation(model, device=device, val_limit=120)
    before_egg, before_leak = _eval_egg_holdout(model, val_egg_imgs, device)
    print(f"\n[Before Refine] FoodSeg={before_fs} | ThaiEggHoldout Hit={before_egg}% (FR-leak={before_leak}%)", flush=True)

    ds = Compiled5GroupDataset(train_rows, is_train=True)
    batch_size = 4
    grad_accum = 3
    epochs = 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=(device.type == "cuda"))

    backbone_params = list(model.segformer.segformer.parameters())
    head_params = (
        list(model.segformer.decode_head.parameters())
        + list(model.shared_proj.parameters())
        + list(model.food_head.parameters())
        + list(model.ratio_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1.2e-4}],
        weight_decay=0.01,
    )
    total_opt_steps = epochs * math.ceil(len(loader) / grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_opt_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    cls_weights = torch.tensor(
        [0.45, 1.0, 1.35, 0.95, 2.2, 1.9, 1.0, 1.1, 1.6, 1.15],
        dtype=torch.float32,
        device=device,
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.03)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_composite = -1e9
    saved = False
    print("\n[3/3] Surgical 2-epoch refinement on RTX 4060 ...", flush=True)
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
                print(f"  Refine Ep {ep}/{epochs} [{step}/{len(loader)}] loss={running_loss/step:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        after_fs = evaluate_validation(model, device=device, val_limit=120)
        after_egg, after_leak = _eval_egg_holdout(model, val_egg_imgs, device)
        score = after_fs["mIoU_pct"] - 1.5 * after_fs["MAE_pct_points"] + 0.2 * after_fs["nonfood_reject_pct"] + 0.25 * after_egg
        print(f"[Refine Ep {ep}] FoodSeg={after_fs} | ThaiEggHoldout Hit={after_egg}% (FR-leak={after_leak}%)", flush=True)

        if (
            after_fs["mIoU_pct"] >= before_fs["mIoU_pct"] - 1.5
            and after_fs["nonfood_reject_pct"] >= 99.0
            and after_egg > before_egg
            and score > best_composite
        ):
            best_composite = score
            torch.save(model.state_dict(), ckpt_path)
            saved = True
            print(f"  -> Saved refined checkpoint to {ckpt_path}", flush=True)

    if saved:
        export_onnx_and_quantize(ckpt_path)
    else:
        print("Refinement did not pass all Pareto retention criteria; keeping original checkpoint.", flush=True)


if __name__ == "__main__":
    main()
