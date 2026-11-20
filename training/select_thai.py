"""Pick THFOOD-50 images for SAM-assisted labelling.

Test images must be ones no model has trained on: THFOOD images used by v2/v3 (compiled_5group) and
v3b (compiled_v3b) are found by matching 16x16 grey thumbnails and excluded from the test pick.

    python training/select_thai.py [test_per_dish=3] [train_per_dish=4] [extra_hard=10]
"""

from __future__ import annotations

import io
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relabel_thai import THFOOD_DISHES  # noqa: E402
from sam_label import CACHE, WORK  # noqa: E402

# Dishes where rice/noodles and protein are mixed or overlap: most labelling effort goes here.
HARD = ["PhatKaphrao", "KaoManGai", "KaoMooDang", "KhaoMooTodGratiem", "KhaoMokGai", "KaiJeowMooSaap",
        "PadThai", "KkaoKlukKaphi", "LarbMoo", "KuaKling", "BooPadPongali", "PadYordMala", "Joke",
        "KorMooYang", "StewedPorkLeg"]


def thumb(b: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(b))
    im.draft("L", (64, 64))
    return np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()


def used_thumbs() -> np.ndarray:
    out = []
    for f in ("compiled_5group_train.parquet", "compiled_v3b_train.parquet"):
        for batch in pq.ParquetFile(os.path.join(CACHE, f)).iter_batches(batch_size=256, columns=["image_bytes", "source"]):
            for r in batch.to_pylist():
                if r["source"] == "thfood50":
                    out.append(thumb(r["image_bytes"]))
    return np.stack(out)


def main(test_per: int = 3, train_per: int = 4, extra_hard: int = 10) -> None:
    used = used_thumbs()
    print("thumbnails of THFOOD images already trained on:", len(used), flush=True)
    by_dish: dict = {}
    for s in range(4):
        idx = 0
        for batch in pq.ParquetFile(os.path.join(CACHE, f"thfood50_train_{s}.parquet")).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                t = thumb(r["image"]["bytes"])
                d = float(np.abs(used - t).mean(1).min())
                by_dish.setdefault(int(r["label"]), []).append({"shard": s, "row": idx, "seen": d < 3.0})
                idx += 1
        print("scanned shard", s, flush=True)
    rng = random.Random(11)
    items, n = [], 0
    for cls, lst in sorted(by_dish.items()):
        dish = THFOOD_DISHES[cls]
        unseen = [x for x in lst if not x["seen"]]
        rng.shuffle(unseen)
        k_train = train_per + (extra_hard if dish in HARD else 0)
        for split, pick in (("test", unseen[:test_per]), ("train", unseen[test_per:test_per + k_train])):
            for x in pick:
                n += 1
                items.append({"id": f"th_{n:05d}", "shard": x["shard"], "row": x["row"], "dish": dish, "split": split})
        print(f"{dish:22s} total {len(lst):4d} unseen {len(unseen):4d}", flush=True)
    os.makedirs(WORK, exist_ok=True)
    json.dump(items, open(os.path.join(WORK, "list.json"), "w", encoding="utf-8"), indent=1)
    print("test", sum(i["split"] == "test" for i in items), "train", sum(i["split"] == "train" for i in items))


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
