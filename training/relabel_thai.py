"""Build cleaner, larger Thai training labels (v3b dataset).

THFOOD-50 has a dish label per image but no pixel masks. The v2-era masks came from an older
teacher that made systematic errors (e.g. minced pork in kaprao labelled "fried rice"). Here the
trained v3 model re-labels the images with:
  * flip test-time augmentation (average of original + mirrored logits),
  * a dish prior: classes that cannot appear in that dish are removed before the argmax
    (kaprao can't contain fruit or sweets; somtam can't contain rice noodles soup ...),
  * an ignore label (255) on pixels whose confidence stays below CONF_MIN, so uncertain
    pixels teach nothing instead of teaching a guess.

Output: datasets_cache/compiled_v3b_train.parquet = FoodSeg103 human masks (rows with rare
classes repeated) + relabelled THFOOD-50 + relabelled Nutrition5k + Tiny-ImageNet non-food.

    python training/relabel_thai.py [per_class=120]
"""

from __future__ import annotations

import io
import os
import random
import sys
from typing import Dict, List, Set

if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_segformer_v3 as T  # noqa: E402

CLASSES = T.CLASSES_V3
CONF_MIN = 0.60
OUT = os.path.join(T.CACHE, "compiled_v3b_train.parquet")

THFOOD_DISHES = [
    "BitterMelonSoup", "BooPadPongali", "CurriedFishCake", "Dumpling", "EggsStewed", "FriedChicken",
    "FriedKale", "FriedMusselPancakes", "GaengJued", "GaengKeawWan", "GaiYang", "GoongObWoonSen",
    "GoongPao", "GrilledQquid", "HoyKraeng", "HoyLaiPrikPao", "Joke", "KaiJeowMooSaap", "KaiThoon",
    "KaoManGai", "KaoMooDang", "KhanomJeenNamYaKati", "KhaoMokGai", "KhaoMooTodGratiem",
    "KhaoNiewMaMuang", "KkaoKlukKaphi", "KorMooYang", "KuaKling", "KuayJab", "KuayTeowReua", "LarbMoo",
    "MassamanGai", "MooSatay", "NamTokMoo", "PadPakBung", "PadPakRuamMit", "PadThai", "PadYordMala",
    "PhatKaphrao", "PorkStickyNoodles", "Roast_duck", "Roast_fish", "Somtam", "SonInLawEggs",
    "StewedPorkLeg", "Suki", "TomKhaGai", "TomYumGoong", "YamWoonSen", "Yentafo",
]

# Classes allowed per dish (background is always allowed). Codes:
# P animal_protein, E egg, T plant_protein, R rice, S other_starch, V vegetable, F fruit,
# M mixed_dish, W sweets, L liquid. Rice is allowed where the dish is normally served on/with rice.
_CODE = {"P": "animal_protein", "E": "egg", "T": "plant_protein", "R": "rice", "S": "other_starch",
         "V": "vegetable", "F": "fruit", "M": "mixed_dish", "W": "sweets", "L": "liquid"}
DISH_ALLOWED_CODES: Dict[str, str] = {
    "BitterMelonSoup": "VPLT", "BooPadPongali": "PEVR", "CurriedFishCake": "PV", "Dumpling": "MPSV",
    "EggsStewed": "EPTLR", "FriedChicken": "PRV", "FriedKale": "VPR", "FriedMusselPancakes": "MPEVS",
    "GaengJued": "LVPTS", "GaengKeawWan": "LPVRS", "GaiYang": "PVR", "GoongObWoonSen": "PSV",
    "GoongPao": "PV", "GrilledQquid": "PV", "HoyKraeng": "PV", "HoyLaiPrikPao": "PV",
    "Joke": "RMLEPV", "KaiJeowMooSaap": "ERV", "KaiThoon": "EVL", "KaoManGai": "RPVL",
    "KaoMooDang": "RPEVL", "KhanomJeenNamYaKati": "SLVPE", "KhaoMokGai": "RMPV", "KhaoMooTodGratiem": "RPVE",
    "KhaoNiewMaMuang": "RFW", "KkaoKlukKaphi": "MRPEVF", "KorMooYang": "PVR", "KuaKling": "PVR",
    "KuayJab": "SLPEV", "KuayTeowReua": "SLPV", "LarbMoo": "PVR", "MassamanGai": "LPVR",
    "MooSatay": "PVSL", "NamTokMoo": "PVR", "PadPakBung": "VPR", "PadPakRuamMit": "VPTR",
    "PadThai": "MSPETV", "PadYordMala": "VPER", "PhatKaphrao": "PREV", "PorkStickyNoodles": "SPVL",
    "Roast_duck": "PRVS", "Roast_fish": "PVS", "Somtam": "VPTR", "SonInLawEggs": "EV",
    "StewedPorkLeg": "PREVL", "Suki": "SPVLE", "TomKhaGai": "LPV", "TomYumGoong": "LPV",
    "YamWoonSen": "SPV", "Yentafo": "SLPVT",
}


def allowed_mask(dish: str) -> np.ndarray:
    names: Set[str] = {"background"} | {_CODE[c] for c in DISH_ALLOWED_CODES[dish]}
    return np.array([n in names for n in CLASSES], dtype=bool)


@torch.no_grad()
def teacher_probs(model, device, im: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    pv, _, (nh, nw) = T.preprocess(im, None, False, random.Random(0))
    x = torch.from_numpy(pv[None]).to(device)
    xf = torch.flip(x, dims=[3])
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        a, _ = model(x)
        b, _ = model(xf)
    # Mirroring the whole padded tensor and mirroring the logits back realigns them exactly.
    logits = (a.float() + torch.flip(b.float(), dims=[3])) / 2
    ch, cw = max(1, round(nh / 4)), max(1, round(nw / 4))
    p = torch.softmax(logits[0, :, :ch, :cw], 0).cpu().numpy()
    return p, (nh, nw)


def label_from_probs(p: np.ndarray, allow: np.ndarray | None) -> np.ndarray:
    if allow is not None:
        p = p * allow[:, None, None]
        p = p / np.maximum(p.sum(0, keepdims=True), 1e-6)
    lab = p.argmax(0).astype(np.uint8)
    lab[p.max(0) < CONF_MIN] = 255
    return lab


def to_full_mask(lab_small: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(lab_small).resize(size, Image.NEAREST), dtype=np.uint8)


def main(per_class: int = 120) -> None:
    device = torch.device("cuda")
    model = T.build_model()
    model.load_state_dict(torch.load(T.CKPT, map_location="cpu", weights_only=True))
    model.to(device).eval()
    rng = random.Random(7)
    rows: List[Dict[str, object]] = []

    # 1. FoodSeg103 human masks from the v3 compile; repeat rows that contain rare classes.
    rare = {CLASSES.index(n) for n in ("egg", "plant_protein", "fruit", "sweets", "liquid")}
    n_fs = n_rep = 0
    for batch in pq.ParquetFile(T.COMPILED_V3).iter_batches(batch_size=256):
        for r in batch.to_pylist():
            if r["source"] != "foodseg103":
                continue
            rows.append(r)
            n_fs += 1
            m = np.asarray(Image.open(io.BytesIO(r["mask_bytes"])))
            if rare & set(np.unique(m).tolist()):
                rows.append(r)
                n_rep += 1
    print(f"FoodSeg103 rows {n_fs} (+{n_rep} repeats with rare classes)", flush=True)

    # 2. THFOOD-50: up to per_class images per dish, relabelled with the dish prior.
    locs: Dict[int, List[tuple[int, int]]] = {}
    for s in range(4):
        for i, lbl in enumerate(pq.read_table(os.path.join(T.CACHE, f"thfood50_train_{s}.parquet"), columns=["label"])["label"].to_pylist()):
            locs.setdefault(int(lbl), []).append((s, i))
    want: Dict[int, Dict[int, int]] = {}
    for cls, lst in locs.items():
        rng.shuffle(lst)
        for s, i in lst[:per_class]:
            want.setdefault(s, {})[i] = cls
    n_th = n_skip = 0
    ignored_frac: List[float] = []
    for s in range(4):
        idx = 0
        for batch in pq.ParquetFile(os.path.join(T.CACHE, f"thfood50_train_{s}.parquet")).iter_batches(batch_size=64, columns=["image"]):
            for r in batch.to_pylist():
                cls = want.get(s, {}).get(idx)
                idx += 1
                if cls is None:
                    continue
                im = T._resize_max(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
                p, _ = teacher_probs(model, device, im)
                lab = label_from_probs(p, allowed_mask(THFOOD_DISHES[cls]))
                food = np.isin(lab, [c for c in range(1, len(CLASSES))])
                if food.mean() < 0.05:
                    n_skip += 1
                    continue
                ignored_frac.append(float((lab == 255).mean()))
                jb, pb = T._encode(im, to_full_mask(lab, im.size))
                rows.append({"image_bytes": jb, "mask_bytes": pb, "is_food": 1.0, "source": "thfood50"})
                n_th += 1
        print(f"  THFOOD shard {s}: {n_th} kept, {n_skip} skipped", flush=True)
    print(f"THFOOD-50 rows {n_th}, mean ignored pixels {np.mean(ignored_frac) * 100:.1f}%", flush=True)

    # 3. Nutrition5k (no dish label): relabel without a prior; 4. non-food as-is.
    n_n5k = n_nf = 0
    for batch in pq.ParquetFile(T.COMPILED_V3).iter_batches(batch_size=128):
        for r in batch.to_pylist():
            if r["source"] == "nonfood":
                rows.append(r)
                n_nf += 1
            elif r["source"] == "nutrition5k":
                im = Image.open(io.BytesIO(r["image_bytes"])).convert("RGB")
                p, _ = teacher_probs(model, device, im)
                lab = label_from_probs(p, None)
                _, pb = T._encode(im, to_full_mask(lab, im.size))
                rows.append({"image_bytes": r["image_bytes"], "mask_bytes": pb, "is_food": 1.0, "source": "nutrition5k"})
                n_n5k += 1
    print(f"Nutrition5k rows {n_n5k}, non-food rows {n_nf}", flush=True)

    pq.write_table(pa.Table.from_pylist(rows), OUT)
    print(f"Wrote {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
