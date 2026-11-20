"""Human-checked Thai labels: SAM 2.1 draws the region boundaries, v3 proposes a class per region,
and a reviewer (a person or Claude looking at the contact sheet) confirms or corrects each region.

    python training/sam_label.py prepare <list.json>     # SAM + v3 proposals + contact sheets
    python training/sam_label.py build                   # labels -> datasets_cache/thai_sam_*.parquet

list.json: [{"id": "th_00012", "shard": 0, "row": 123, "dish": "PhatKaphrao", "split": "test"}, ...]
Review file training/thai_labels.json: {"th_00012": {"5": "P", "7": "R", "rest": "b"}, ...}
  keys are region numbers on the sheet; values are class codes (CODES below). Regions not listed keep
  the v3 proposal. "rest" = pixels SAM left unassigned (default: v3 proposal where confident, else ignore).
  {"skip": true} drops the image (bad photo / not the dish).
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few THFOOD JPEGs are cut short

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "datasets_cache")
WORK = os.path.join(CACHE, "thai_sam")
SHEETS = os.path.join(WORK, "sheets")
LABELS = os.path.join(ROOT, "training", "thai_labels.json")
CLASSES = ["background", "rice", "other_starch", "animal_protein", "egg", "plant_protein",
           "vegetable", "fruit", "mixed_dish", "sweets", "liquid"]
CODES = {"b": 0, "R": 1, "S": 2, "P": 3, "E": 4, "T": 5, "V": 6, "F": 7, "M": 8, "W": 9, "L": 10, "X": 255}
CODE_OF = {v: k for k, v in CODES.items()}
COLORS = [(40, 40, 40), (245, 245, 220), (230, 170, 60), (200, 50, 50), (255, 220, 0), (150, 100, 200),
          (40, 170, 60), (255, 120, 180), (120, 80, 40), (255, 0, 255), (60, 140, 230)]
MAX_SIDE = 1024
MIN_REGION = 0.002  # regions under 0.2% of the image merge into "rest"


def load_image(shard: int, row: int) -> Image.Image:
    t = pq.ParquetFile(os.path.join(CACHE, f"thfood50_train_{shard}.parquet"))
    rg_start = 0
    for g in range(t.num_row_groups):
        n = t.metadata.row_group(g).num_rows
        if row < rg_start + n:
            r = t.read_row_group(g, columns=["image"]).slice(row - rg_start, 1).to_pylist()[0]
            im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            s = MAX_SIDE / max(im.size)
            return im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR) if s < 1 else im
        rg_start += n
    raise IndexError(row)


def region_map(masks: List[np.ndarray]) -> np.ndarray:
    """Non-overlapping region ids (1..K); 0 = unassigned. Larger masks first so finer masks win."""
    h, w = masks[0].shape if masks else (1, 1)
    reg = np.zeros((h, w), dtype=np.int32)
    k = 0
    for m in sorted(masks, key=lambda m: -int(m.sum())):
        k += 1
        reg[m] = k
    # Renumber, dropping regions too small to matter.
    out = np.zeros_like(reg)
    nid = 0
    for rid in np.unique(reg):
        if rid == 0:
            continue
        sel = reg == rid
        if sel.mean() < MIN_REGION:
            continue
        nid += 1
        out[sel] = nid
    return out


def fill_rest(reg: np.ndarray, v3lab: np.ndarray) -> np.ndarray:
    """SAM's automatic mode leaves large food areas unassigned (e.g. a pile of meat). Split those
    pixels into connected pieces of the same v3 class so the reviewer sees and checks them too."""
    from scipy import ndimage

    reg = reg.copy()
    nid = int(reg.max())
    rest = reg == 0
    for c in np.unique(v3lab[rest]):
        comp, n = ndimage.label(rest & (v3lab == c))
        for k in range(1, n + 1):
            sel = comp == k
            if sel.mean() >= MIN_REGION * 2:
                nid += 1
                reg[sel] = nid
    return reg


def _put(d: ImageDraw.ImageDraw, font, sel: np.ndarray, txt: str, W: int, H: int) -> None:
    """Draw txt at the pixel deepest inside sel (largest distance to its border), so a label can never
    sit on the rim of a large region next to a small one."""
    from scipy import ndimage

    if not sel.any():
        return
    dist = ndimage.distance_transform_edt(np.pad(sel, 1))[1:-1, 1:-1]
    y0, x0 = np.unravel_index(int(dist.argmax()), dist.shape)
    x = min(max(int(x0) - 4 * len(txt), 1), W - 9 * len(txt) - 2)
    y = min(max(int(y0) - 8, 1), H - 18)
    d.rectangle([x - 1, y - 1, x + 9 * len(txt), y + 16], fill=(0, 0, 0))
    d.text((x + 1, y), txt, fill=(255, 255, 0), font=font)


def render_final(im: Image.Image, lab: np.ndarray, title: str, W: int = 480) -> Image.Image:
    """Original | final label map, letters at each sizeable piece. Used to check finished labels."""
    from scipy import ndimage

    s = W / im.width
    H = round(im.height * s)
    a = im.resize((W, H), Image.BILINEAR)
    labs = np.asarray(Image.fromarray(lab).resize((W, H), Image.NEAREST))
    col = np.zeros((H, W, 3), dtype=np.uint8)
    for c in np.unique(labs):
        col[labs == c] = COLORS[c] if c != 255 else (128, 128, 128)
    over = (np.asarray(a, dtype=np.float32) * 0.4 + col * 0.6).astype(np.uint8)
    b = Image.fromarray(over)
    d = ImageDraw.Draw(b)
    try:
        font = ImageFont.truetype("arialbd.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for c in np.unique(labs):
        comp, n = ndimage.label(labs == c)
        for k in range(1, n + 1):
            sel = comp == k
            if sel.mean() > 0.01:
                _put(d, font, sel, CODE_OF[int(c)], W, H)
    out = Image.new("RGB", (W * 2 + 8, H + 22), (255, 255, 255))
    out.paste(a, (0, 22))
    out.paste(b, (W + 8, 22))
    ImageDraw.Draw(out).text((4, 3), title, fill=(0, 0, 0), font=font)
    return out


def check(ids: List[str]) -> None:
    """Write check sheets (two images stacked per file) of the final labels for the given ids."""
    labels = json.load(open(LABELS, encoding="utf-8"))
    os.makedirs(os.path.join(WORK, "check"), exist_ok=True)
    tiles = []
    for iid in ids:
        rev = labels.get(iid, {})
        im = Image.open(os.path.join(WORK, f"{iid}.jpg")).convert("RGB")
        lab = np.zeros((im.height, im.width), np.uint8) if rev.get("skip") else final_mask(iid, rev)
        tiles.append(render_final(im, lab, iid + (" SKIP" if rev.get("skip") else "")))
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        w = max(t.width for t in pair)
        sheet = Image.new("RGB", (w, sum(t.height for t in pair) + 6 * (len(pair) - 1)), (255, 255, 255))
        y = 0
        for t in pair:
            sheet.paste(t, (0, y))
            y += t.height + 6
        sheet.save(os.path.join(WORK, "check", f"{ids[i]}.jpg"), quality=85)


def resheet() -> None:
    """Re-render labelling sheets from saved npz files (after a rendering change)."""
    items = {it["id"]: it for it in json.load(open(os.path.join(WORK, "list.json"), encoding="utf-8"))}
    for iid, it in items.items():
        npz = os.path.join(WORK, f"{iid}.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        prop = {int(k): int(v) for k, v in z["prop"]}
        im = Image.open(os.path.join(WORK, f"{iid}.jpg")).convert("RGB")
        render_sheet(im, z["reg"], prop, f"{iid}  {it['dish']}  ({it['split']})").save(
            os.path.join(SHEETS, f"{iid}.jpg"), quality=85)


def render_sheet(im: Image.Image, reg: np.ndarray, prop: Dict[int, int], title: str) -> Image.Image:
    W = 640
    s = W / im.width
    H = round(im.height * s)
    a = im.resize((W, H), Image.BILINEAR)
    regs = np.asarray(Image.fromarray(reg.astype(np.int32)).resize((W, H), Image.NEAREST))
    col = np.zeros((H, W, 3), dtype=np.uint8)
    for rid, c in prop.items():
        col[regs == rid] = COLORS[c] if c != 255 else (128, 128, 128)
    base = np.asarray(a, dtype=np.float32)
    over = (base * 0.45 + col * 0.55).astype(np.uint8)
    edge = np.zeros((H, W), dtype=bool)
    edge[1:, :] |= regs[1:, :] != regs[:-1, :]
    edge[:, 1:] |= regs[:, 1:] != regs[:, :-1]
    over[edge] = (255, 255, 255)
    b = Image.fromarray(over)
    d = ImageDraw.Draw(b)
    try:
        font = ImageFont.truetype("arialbd.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for rid, c in prop.items():
        _put(d, font, regs == rid, f"{rid}{CODE_OF[c]}", W, H)
    sheet = Image.new("RGB", (W * 2 + 10, H + 24), (255, 255, 255))
    sheet.paste(a, (0, 24))
    sheet.paste(b, (W + 10, 24))
    ImageDraw.Draw(sheet).text((4, 4), title, fill=(0, 0, 0), font=font)
    return sheet


def prepare(list_path: str) -> None:
    import torch
    from transformers import pipeline
    import train_segformer_v3 as T

    os.makedirs(SHEETS, exist_ok=True)
    items = json.load(open(list_path, encoding="utf-8"))
    sam = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)
    dev = torch.device("cuda")
    model = T.build_model()
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "segformer_b2_v3.pt"), map_location="cpu", weights_only=True))
    model.to(dev).eval()
    import random
    for n, it in enumerate(items):
        npz = os.path.join(WORK, f"{it['id']}.npz")
        if os.path.exists(npz):
            continue
        im = load_image(it["shard"], it["row"])
        out = sam(im, points_per_batch=64, pred_iou_thresh=0.7, stability_score_thresh=0.85)
        masks = [np.asarray(m.cpu() if hasattr(m, "cpu") else m).astype(bool) for m in out["masks"]]
        reg = region_map(masks) if masks else np.zeros((im.height, im.width), np.int32)
        # v3 probabilities at full image resolution.
        pv, _, (nh, nw) = T.preprocess(im, None, False, random.Random(0))
        with torch.no_grad(), torch.amp.autocast("cuda"):
            lg, _ = model(torch.from_numpy(pv[None]).to(dev))
        ch, cw = max(1, round(nh / 4)), max(1, round(nw / 4))
        p = torch.softmax(lg.float()[0, :, :ch, :cw], 0)[None]
        p = torch.nn.functional.interpolate(p, size=(im.height, im.width), mode="bilinear", align_corners=False)[0].cpu().numpy()
        v3lab, v3conf = p.argmax(0).astype(np.uint8), p.max(0)
        reg = fill_rest(reg, v3lab)
        prop: Dict[int, int] = {}
        for rid in range(1, int(reg.max()) + 1):
            sel = reg == rid
            if sel.any():
                prop[rid] = int(np.bincount(v3lab[sel], minlength=len(CLASSES)).argmax())
        np.savez_compressed(npz, reg=reg, v3lab=v3lab, v3conf=v3conf.astype(np.float16),
                            prop=np.array([[k, v] for k, v in prop.items()], dtype=np.int32).reshape(-1, 2))
        im.save(os.path.join(WORK, f"{it['id']}.jpg"), quality=92)
        render_sheet(im, reg, prop, f"{it['id']}  {it['dish']}  ({it['split']})").save(
            os.path.join(SHEETS, f"{it['id']}.jpg"), quality=85)
        if n % 20 == 0:
            print(f"{n + 1}/{len(items)} {it['id']} regions={len(prop)}", flush=True)


def repropose(ckpt: str) -> None:
    """Replace class proposals (keeping SAM regions) with a newer model's, for images not yet reviewed.
    A better proposer means fewer corrections. Test images are left alone."""
    import random

    import torch
    import train_segformer_v3 as T

    labels = json.load(open(LABELS, encoding="utf-8"))
    items = [it for it in json.load(open(os.path.join(WORK, "list.json"), encoding="utf-8"))
             if it["split"] == "train" and it["id"] not in labels]
    dev = torch.device("cuda")
    model = T.build_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(dev).eval()
    for it in items:
        npz = os.path.join(WORK, f"{it['id']}.npz")
        z = dict(np.load(npz))
        im = Image.open(os.path.join(WORK, f"{it['id']}.jpg")).convert("RGB")
        pv, _, (nh, nw) = T.preprocess(im, None, False, random.Random(0))
        with torch.no_grad():
            lg, _ = model(torch.from_numpy(pv[None]).to(dev))
        ch, cw = max(1, round(nh / 4)), max(1, round(nw / 4))
        p = torch.softmax(lg.float()[0, :, :ch, :cw], 0)[None]
        p = torch.nn.functional.interpolate(p, size=(im.height, im.width), mode="bilinear", align_corners=False)[0].cpu().numpy()
        lab, conf = p.argmax(0).astype(np.uint8), p.max(0)
        reg = z["reg"]
        prop = {rid: int(np.bincount(lab[reg == rid], minlength=len(CLASSES)).argmax())
                for rid in range(1, int(reg.max()) + 1) if (reg == rid).any()}
        np.savez_compressed(npz, reg=reg, v3lab=lab, v3conf=conf.astype(np.float16),
                            prop=np.array([[k, v] for k, v in prop.items()], dtype=np.int32).reshape(-1, 2))
        render_sheet(im, reg, prop, f"{it['id']}  {it['dish']}  ({it['split']})").save(
            os.path.join(SHEETS, f"{it['id']}.jpg"), quality=85)
    print("re-proposed", len(items))


def final_mask(item_id: str, review: Dict[str, str]) -> np.ndarray:
    z = np.load(os.path.join(WORK, f"{item_id}.npz"))
    reg, v3lab, v3conf = z["reg"], z["v3lab"], z["v3conf"].astype(np.float32)
    prop = {int(k): int(v) for k, v in z["prop"]}
    lab = np.where(v3conf >= 0.6, v3lab, 255).astype(np.uint8)  # unassigned pixels
    if "rest" in review:
        lab[reg == 0] = CODES[review["rest"]]
    remap = {CODES[a]: CODES[b] for a, b in review.get("map", {}).items()}  # e.g. {"b": "P"}
    v3pix = np.where(v3conf >= 0.6, v3lab, 255).astype(np.uint8)
    for rid, c in prop.items():
        code = review.get(str(rid))
        sel = reg == rid
        if code == "v3":  # SAM region spans food and non-food: keep v3's own per-pixel labels
            lab[sel] = v3pix[sel]
        else:
            lab[sel] = CODES[code] if code is not None else remap.get(c, c)
    # "boxes": [[x0, y0, x1, y1, code], ...] in contact-sheet pixels of the right panel (which starts
    # at x=650, y=24 and is 640 px wide) for spots where SAM merged two different foods.
    s = 640 / reg.shape[1]
    # An optional 6th element limits the box to pixels currently labelled with that code.
    for x0, y0, x1, y1, code, *only in review.get("boxes", []):
        c0, c1 = max(0, int((x0 - 650) / s)), int((x1 - 650) / s)
        r0, r1 = max(0, int((y0 - 24) / s)), int((y1 - 24) / s)
        win = lab[r0:r1, c0:c1]
        if only:
            win[win == CODES[only[0]]] = CODES[code]
        else:
            win[:] = CODES[code]
    return lab


def build() -> None:
    import train_segformer_v3 as T

    items = {it["id"]: it for it in json.load(open(os.path.join(WORK, "list.json"), encoding="utf-8"))}
    labels = json.load(open(LABELS, encoding="utf-8"))
    rows = {"test": [], "train": []}
    for iid, rev in labels.items():
        if rev.get("skip"):
            continue
        it = items[iid]
        im = Image.open(os.path.join(WORK, f"{iid}.jpg")).convert("RGB")
        lab = final_mask(iid, rev)
        jb, pb = T._encode(im, lab)
        rows[it["split"]].append({"image_bytes": jb, "mask_bytes": pb, "is_food": 1.0, "source": "thai_sam",
                                  "id": iid, "dish": it["dish"]})
    for split, r in rows.items():
        if r:
            pq.write_table(pa.Table.from_pylist(r), os.path.join(CACHE, f"thai_sam_{split}.parquet"))
        print(split, len(r))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2])
    elif sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "check":
        check(sys.argv[2:])
    elif sys.argv[1] == "repropose":
        repropose(sys.argv[2])
    elif sys.argv[1] == "resheet":
        resheet()
