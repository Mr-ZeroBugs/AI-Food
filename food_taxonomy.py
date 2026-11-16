"""Shared FoodSeg103 / THFOOD-50 / Food-101 taxonomy, Thai context fusion, and pure-NumPy post-processing."""

from __future__ import annotations

from collections import deque
import gzip
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def extract_gz_to_temp(src_gz: str, dst_path: str) -> str:
    """Decompress a `.gz` asset to `dst_path` atomically once (re-extract if `.gz` is newer)."""
    if (
        os.path.exists(dst_path)
        and os.path.getsize(dst_path) > 1024
        and os.path.exists(src_gz)
        and os.path.getmtime(dst_path) >= os.path.getmtime(src_gz)
    ):
        return dst_path
    if not os.path.exists(src_gz):
        raise FileNotFoundError(f"Compressed asset missing: {src_gz}")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = f"{dst_path}.tmp.{os.getpid()}"
    with gzip.open(src_gz, "rb") as fin, open(tmp_path, "wb") as fout:
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
    os.replace(tmp_path, dst_path)
    return dst_path


FOOD101_THAI_NAMES: Dict[str, str] = {
    "fried_rice": "ข้าวผัด",
    "pad_thai": "ผัดไทย",
    "pho": "ก๋วยเตี๋ยวน้ำใส/เฝอ",
    "ramen": "ราเม็ง/บะหมี่น้ำ",
    "dumplings": "เกี๊ยวซ่า/ขนมจีบ",
    "gyoza": "เกี๊ยวซ่า",
    "spring_rolls": "ปอเปี๊ยะทอด",
    "sushi": "ซูชิ",
    "sashimi": "ซาชิมิ/ปลาดิบ",
    "bibimbap": "ข้าวยำเกาหลี (บิบิมบับ)",
    "peking_duck": "เป็ดปักกิ่ง/เป็ดย่าง",
    "omelette": "ไข่เจียว/ออมเล็ต",
    "grilled_salmon": "สเต๊กปลาแซลมอนย่าง",
    "steak": "สเต๊กเนื้อ",
    "pork_chop": "พอร์คชอป",
    "chicken_wings": "ปีกไก่ทอด",
    "chicken_curry": "แกงกะหรี่ไก่",
    "caesar_salad": "สลัดผัก/ซีซาร์สลัด",
    "greek_salad": "สลัดผักรวม",
    "seaweed_salad": "ยำสาหร่าย",
    "french_fries": "เฟรนช์ฟรายส์",
    "hamburger": "แฮมเบอร์เกอร์",
    "pizza": "พิซซ่า",
    "spaghetti_bolognese": "สปาเก็ตตี้ซอสเนื้อ",
    "spaghetti_carbonara": "สปาเก็ตตี้คาโบนาร่า",
    "paella": "ข้าวผัดซีฟู้ด/ปาเอยา",
    "fried_calamari": "ปลาหมึกทอด",
    "edamame": "ถั่วแระญี่ปุ่น",
}

THFOOD50_THAI_NAMES: Dict[str, str] = {
    "BitterMelonSoup": "ต้มจืดมะระ",
    "BooPadPongali": "ปูผัดผงกะหรี่",
    "CurriedFishCake": "ทอดมันปลากราย",
    "Dumpling": "ขนมจีบ/เกี๊ยว",
    "EggsStewed": "ไข่พะโล้",
    "FriedChicken": "ไก่ทอด",
    "FriedKale": "ผัดคะน้า",
    "FriedMusselPancakes": "หอยทอด",
    "GaengJued": "แกงจืด",
    "GaengKeawWan": "แกงเขียวหวาน",
    "GaiYang": "ไก่ย่าง",
    "GoongObWoonSen": "กุ้งอบวุ้นเส้น",
    "GoongPao": "กุ้งเผา",
    "GrilledQquid": "ปลาหมึกย่าง",
    "HoyKraeng": "หอยแครงลวก",
    "HoyLaiPrikPao": "หอยลายผัดพริกเผา",
    "Joke": "โจ๊ก",
    "KaiJeowMooSaap": "ไข่เจียวหมูสับ",
    "KaiThoon": "ไข่ตุ๋น",
    "KaoManGai": "ข้าวมันไก่",
    "KaoMooDang": "ข้าวหมูแดง",
    "KhanomJeenNamYaKati": "ขนมจีนน้ำยากะทิ",
    "KhaoMokGai": "ข้าวหมกไก่",
    "KhaoMooTodGratiem": "ข้าวหมูทอดกระเทียม",
    "KhaoNiewMaMuang": "ข้าวเหนียวมะม่วง",
    "KkaoKlukKaphi": "ข้าวคลุกกะปิ",
    "KorMooYang": "คอหมูย่าง",
    "KuaKling": "คั่วกลิ้ง",
    "KuayJab": "กวยจั๊บ",
    "KuayTeowReua": "ก๋วยเตี๋ยวเรือ",
    "LarbMoo": "ลาบหมู",
    "MassamanGai": "แกงมัสมั่นไก่",
    "MooSatay": "หมูสะเต๊ะ",
    "NamTokMoo": "น้ำตกหมู",
    "PadPakBung": "ผัดผักบุ้ง",
    "PadPakRuamMit": "ผัดผักรวมมิตร",
    "PadThai": "ผัดไทย",
    "PadYordMala": "ผัดยอดมะระ",
    "PhatKaphrao": "ผัดกะเพรา",
    "PorkStickyNoodles": "ราดหน้าหมู",
    "Roast_duck": "เป็ดย่าง",
    "Roast_fish": "ปลาเผา",
    "Somtam": "ส้มตำ",
    "SonInLawEggs": "ไข่ลูกเขย",
    "StewedPorkLeg": "ข้าวขาหมู",
    "Suki": "สุกี้",
    "TomKhaGai": "ต้มข่าไก่",
    "TomYumGoong": "ต้มยำกุ้ง",
    "YamWoonSen": "ยำวุ้นเส้น",
    "Yentafo": "เย็นตาโฟ",
}

FOODSEG103_ID2LABEL: Dict[int, str] = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients",
}

FOODSEG103_THAI_NAMES: Dict[int, str] = {
    1: "ลูกอม/ของหวาน",
    2: "ทาร์ตไข่",
    3: "เฟรนช์ฟรายส์",
    4: "ช็อกโกแลต",
    5: "บิสกิต/คุกกี้",
    6: "ป๊อปคอร์น",
    7: "พุดดิ้ง",
    8: "ไอศกรีม",
    9: "ชีส/เนย",
    10: "เค้ก",
    11: "ไวน์/เครื่องดื่ม",
    12: "มิลค์เชค",
    13: "กาแฟ",
    14: "น้ำผลไม้",
    15: "นม",
    16: "ชา",
    17: "อัลมอนด์",
    18: "ถั่วแดง",
    19: "เม็ดมะม่วงหิมพานต์",
    20: "แครนเบอร์รีอบแห้ง",
    21: "ถั่วเหลือง",
    22: "วอลนัท",
    23: "ถั่วลิสง",
    24: "ไข่",
    25: "แอปเปิล",
    26: "อินทผลัม",
    27: "แอปริคอต",
    28: "อะโวคาโด",
    29: "กล้วย",
    30: "สตรอว์เบอร์รี",
    31: "เชอร์รี",
    32: "บลูเบอร์รี",
    33: "ราสป์เบอร์รี",
    34: "มะม่วง",
    35: "มะกอก",
    36: "พีช",
    37: "มะนาว/เลมอน",
    38: "ลูกแพร์",
    39: "มะเดื่อ",
    40: "สับปะรด",
    41: "องุ่น",
    42: "กีวี",
    43: "เมลอน",
    44: "ส้ม",
    45: "แตงโม",
    46: "สเต๊ก/เนื้อวัว",
    47: "เนื้อหมู/หมูสับ",
    48: "เนื้อไก่/เป็ด",
    49: "ไส้กรอก",
    50: "เนื้อสัตว์ผัด/ทอด",
    51: "เนื้อแกะ",
    52: "ซอส/น้ำราด",
    53: "ปู",
    54: "เนื้อสัตว์ผัด/เนื้อปลา",
    55: "หอย/อาหารทะเล",
    56: "กุ้ง",
    57: "น้ำซุป/แกง",
    58: "ขนมปัง",
    59: "ข้าวโพด",
    60: "แฮมเบอร์เกอร์",
    61: "พิซซ่า",
    62: "ซาลาเปา/หมั่นโถว",
    63: "เกี๊ยว",
    64: "พาสต้า",
    65: "เส้นก๋วยเตี๋ยว/บะหมี่",
    66: "ข้าวสวย",
    67: "พาย",
    68: "เต้าหู้",
    69: "มะเขือยาว",
    70: "มันฝรั่ง",
    71: "กระเทียม",
    72: "ดอกกะหล่ำ",
    73: "มะเขือเทศ",
    74: "สาหร่ายเคลป์",
    75: "สาหร่ายทะเล",
    76: "ต้นหอม",
    77: "ผักกาดกวางตุ้ง",
    78: "ขิง",
    79: "กระเจี๊ยบเขียว",
    80: "ผักกาดหอม",
    81: "ฟักทอง",
    82: "แตงกวา",
    83: "หัวไชเท้า",
    84: "แครอท",
    85: "หน่อไม้ฝรั่ง",
    86: "หน่อไม้",
    87: "บรอกโคลี",
    88: "ขึ้นฉ่ายฝรั่ง",
    89: "ใบกะเพรา/ผักชี/สมุนไพร",
    90: "ถั่วลันเตา",
    91: "กะหล่ำปลี",
    92: "ถั่วงอก",
    93: "หัวหอม",
    94: "พริก/พริกไทย",
    95: "ถั่วเขียว/ถั่วฝักยาว",
    96: "ถั่วแขก",
    97: "เห็ดออรินจิ",
    98: "เห็ดหอม",
    99: "เห็ดเข็มทอง",
    100: "เห็ดนางรม",
    101: "เห็ดแชมปิญอง",
    102: "สลัดผัก",
    103: "วัตถุดิบอื่นๆ",
}


def _build_foodseg_to_local_map() -> Dict[int, int]:
    """Map FoodSeg103 class IDs (0..103) into the 10 local nutrition classes."""
    mapping: Dict[int, int] = {0: 0}
    for cid in (58, 59, 62, 66, 70):
        mapping[cid] = 1
    for cid in (64, 65):
        mapping[cid] = 2
    for cid in (1, 2, 3, 5, 6, 7, 10, 60, 61, 63, 67):
        mapping[cid] = 9
    for cid in (12, 15, 17, 18, 19, 21, 22, 23, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 68):
        mapping[cid] = 3
    mapping[24] = 5
    mapping[4] = 3
    mapping[8] = 3
    mapping[9] = 3
    for cid in (74, 75, 76, 77, 80, 85, 87, 88, 89, 90, 91, 92, 95, 96, 102):
        mapping[cid] = 6
    for cid in (69, 71, 72, 73, 78, 79, 81, 82, 83, 84, 86, 93, 94, 97, 98, 99, 100, 101):
        mapping[cid] = 7
    for cid in range(25, 46):
        mapping[cid] = 8
    mapping[20] = 8
    mapping[14] = 8
    for cid in (11, 13, 16, 52, 57, 103):
        mapping[cid] = 0
    return mapping


FOODSEG103_TO_LOCAL_10CLASS: Dict[int, int] = _build_foodseg_to_local_map()


def _unshift_mericberktas_id(pred_id: int) -> int:
    """Decode shifted label indices from mericberktas/foodseg-models back to canonical IDs."""
    if pred_id == 0:
        return 0
    if 1 <= pred_id <= 12:
        return pred_id + 91
    if pred_id == 13:
        return 91
    if 14 <= pred_id <= 103:
        return pred_id - 13
    return 0


_UNSHIFT_LUT = np.array([_unshift_mericberktas_id(i) for i in range(256)], dtype=np.uint8)

_THAI_PROTEIN_CONTEXT: Dict[str, Tuple[str, str]] = {
    "PhatKaphrao": ("หมูสับ/เนื้อสัตว์ผัดกะเพรา", "minced_pork_kaphrao"),
    "KaoMooDang": ("หมูแดง/หมูกรอบ", "roast_pork_thai"),
    "KhaoMooTodGratiem": ("หมูทอดกระเทียม", "garlic_pork"),
    "KorMooYang": ("คอหมูย่าง", "grilled_pork_neck"),
    "LarbMoo": ("หมูสับลาบ", "spicy_minced_pork"),
    "NamTokMoo": ("หมูย่างน้ำตก", "grilled_pork_salad"),
    "StewedPorkLeg": ("ขาหมูพะโล้", "stewed_pork_leg"),
    "KuaKling": ("เนื้อสัตว์คั่วกลิ้ง", "dry_curry_pork"),
    "MooSatay": ("หมูสะเต๊ะ", "pork_satay"),
    "PorkStickyNoodles": ("เนื้อหมู (ราดหน้า)", "pork_in_gravy"),
    "KaoManGai": ("เนื้อไก่ต้ม (ข้าวมันไก่)", "steamed_chicken"),
    "GaiYang": ("ไก่ย่าง", "grilled_chicken"),
    "FriedChicken": ("ไก่ทอด", "fried_chicken"),
    "KhaoMokGai": ("เนื้อไก่ (ข้าวหมกไก่)", "spiced_chicken"),
    "TomKhaGai": ("เนื้อไก่ (ต้มข่าไก่)", "chicken_tomkha"),
    "MassamanGai": ("เนื้อไก่ (มัสมั่น)", "chicken_massaman"),
}

_THFOOD_SPICY_CHILI_DISHES = {
    "PhatKaphrao",
    "KuaKling",
    "LarbMoo",
    "NamTokMoo",
    "Somtam",
    "YamWoonSen",
    "PadPakBung",
    "HoyLaiPrikPao",
    "TomYumGoong",
}

_KHAOPAD_CODES = {"KhaoPadGoong", "KhaoPadPoo", "KhaoPadSaiKrok", "KhaoPad", "fried_rice"}


def _apply_thai_culinary_context(
    cid: int,
    eng_name: str,
    thai_name: str,
    thai_dish_info: Dict[str, object],
) -> Tuple[str, str]:
    """Refine generic FoodSeg103 ingredient labels using Tri-Model dish context."""
    conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    dish_code = str(thai_dish_info.get("top1_code", ""))
    if conf < 60.0 or not dish_code:
        return eng_name, thai_name

    if dish_code == "KaoManGai":
        if cid == 48:
            return "steamed_chicken", "เนื้อไก่ต้ม (ข้าวมันไก่)"
        if cid == 66:
            return "oily_chicken_rice", "ข้าวมัน (หุงน้ำซุปไก่)"
    if dish_code in _KHAOPAD_CODES:
        if cid == 66:
            return "fried_rice_egg", "ข้าวผัดไข่"
        if cid == 56:
            return "shrimp", "กุ้งสด"
        if cid == 37:
            return "lime", "มะนาวซีก"

    if 46 <= cid <= 56 and dish_code in _THAI_PROTEIN_CONTEXT:
        ctx_thai, ctx_eng = _THAI_PROTEIN_CONTEXT[dish_code]
        return ctx_eng, ctx_thai
    if dish_code == "PhatKaphrao":
        if cid == 89:
            return "holy_basil", "ใบกะเพรา"
        if cid == 94:
            return "thai_chili", "พริกจินดาแดง"
    return eng_name, thai_name


def _dilate_bool_5x5(mask_bool: np.ndarray) -> np.ndarray:
    pad = np.pad(mask_bool, 2, mode="constant", constant_values=False)
    h, w = mask_bool.shape
    out = np.zeros_like(mask_bool, dtype=bool)
    for dy in range(5):
        for dx in range(5):
            out |= pad[dy : dy + h, dx : dx + w]
    return out


def _components_touching_seed(
    target: np.ndarray,
    seed_dilated: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Find connected components in `target` meeting `min_area` and touching `seed_dilated` (pure NumPy)."""
    seeds = target & seed_dilated
    if not np.any(seeds):
        return np.zeros_like(target, dtype=bool)

    h, w = target.shape
    # Work on a <= 192px grid for O(1ms) pure-Python BFS when input is large
    scale = min(1.0, 192.0 / max(h, w))
    if scale < 1.0:
        sw, sh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        t_small = np.asarray(
            Image.fromarray(target.astype(np.uint8)).resize((sw, sh), Image.NEAREST),
            dtype=bool,
        )
        s_small = np.asarray(
            Image.fromarray(seeds.astype(np.uint8)).resize((sw, sh), Image.NEAREST),
            dtype=bool,
        )
        area_scale = (sw / w) * (sh / h)
        min_small = max(2, int(round(min_area * area_scale)))
    else:
        t_small, s_small = target, seeds
        sw, sh = w, h
        min_small = min_area

    visited = np.zeros((sh, sw), dtype=bool)
    keep_small = np.zeros((sh, sw), dtype=bool)
    sy_xs = np.argwhere(s_small)
    for sy_np, sx_np in sy_xs:
        sy, sx = int(sy_np), int(sx_np)
        if visited[sy, sx]:
            continue
        q: deque[Tuple[int, int]] = deque([(sy, sx)])
        visited[sy, sx] = True
        comp: List[Tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for ny in range(max(0, cy - 1), min(sh, cy + 2)):
                for nx in range(max(0, cx - 1), min(sw, cx + 2)):
                    if not visited[ny, nx] and t_small[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        if len(comp) >= min_small:
            for cy, cx in comp:
                keep_small[cy, cx] = True

    if not np.any(keep_small):
        return np.zeros_like(target, dtype=bool)
    if scale < 1.0:
        keep_full = np.asarray(
            Image.fromarray(keep_small.astype(np.uint8)).resize((w, h), Image.NEAREST),
            dtype=bool,
        )
        return target & keep_full
    return keep_small


def merge_shrimp_body_tail_components(raw_mask: np.ndarray) -> None:
    """Merge FoodSeg103 'sausage' (49) + attached red tail (94) components into 'shrimp' (56)."""
    m49 = raw_mask == 49
    m94 = raw_mask == 94
    if not np.any(m49) or not np.any(m94):
        return

    dilated_94 = _dilate_bool_5x5(m94)
    merged_bodies = _components_touching_seed(m49, dilated_94, min_area=50)
    if not np.any(merged_bodies):
        return

    dilated_bodies = _dilate_bool_5x5(merged_bodies)
    merged_tails = _components_touching_seed(m94, dilated_bodies, min_area=15)
    raw_mask[merged_bodies | merged_tails] = 56


def fuse_multimodel_context(
    raw_mask: np.ndarray,
    raw_top2_mask: np.ndarray,
    thai_dish_info: Dict[str, object],
    food101_info: Dict[str, object],
) -> Dict[str, object]:
    """Fuse FoodSeg103 Top-2 queries + THFOOD-50 + Food-101 without blind overwrites."""
    total_fg = int(np.count_nonzero(raw_mask > 0))
    carb_ratio = int(np.count_nonzero(np.isin(raw_mask, [65, 66]))) / max(1, total_fg)

    th_code = str(thai_dish_info.get("top1_code", ""))
    th_conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    f101_code = str(food101_info.get("top1_code", ""))
    f101_conf = float(food101_info.get("top1_conf_pct", 0.0))

    if (f101_code in ("fried_rice", "paella") and f101_conf >= 45.0) or carb_ratio >= 0.30:
        merge_shrimp_body_tail_components(raw_mask)

    existing_top3 = list(thai_dish_info.get("top3", []))
    if (
        f101_code == "fried_rice"
        and f101_conf >= 45.0
        and th_conf < 65.0
        and carb_ratio >= 0.20
    ):
        if int(np.count_nonzero(raw_mask == 56)) >= 400:
            code, thai_name = "KhaoPadGoong", "ข้าวผัดกุ้ง"
        elif int(np.count_nonzero(raw_mask == 53)) >= 400:
            code, thai_name = "KhaoPadPoo", "ข้าวผัดปู"
        elif int(np.count_nonzero(raw_mask == 49)) >= 400:
            code, thai_name = "KhaoPadSaiKrok", "ข้าวผัดไส้กรอก"
        else:
            code, thai_name = "KhaoPad", "ข้าวผัดไข่/หมู"

        promoted = {"code": code, "thai_name": thai_name, "conf_pct": f101_conf, "source": "Food-101"}
        thai_dish_info = {
            "top1_code": code,
            "top1_thai": thai_name,
            "top1_conf_pct": f101_conf,
            "top3": [promoted] + existing_top3[:2],
        }
    elif (
        th_conf < 55.0
        and f101_conf >= 55.0
        and f101_code in FOOD101_THAI_NAMES
        and (f101_code != "fried_rice" or carb_ratio >= 0.20)
    ):
        thai_name = FOOD101_THAI_NAMES[f101_code]
        promoted = {"code": f101_code, "thai_name": thai_name, "conf_pct": f101_conf, "source": "Food-101"}
        thai_dish_info = {
            "top1_code": f101_code,
            "top1_thai": thai_name,
            "top1_conf_pct": f101_conf,
            "top3": [promoted] + existing_top3[:2],
        }

    if (
        str(thai_dish_info.get("top1_code", "")) == "KaoManGai"
        and float(thai_dish_info.get("top1_conf_pct", 0.0)) >= 60.0
    ):
        top2_verified = np.isin(raw_mask, [52, 93, 103]) & np.isin(
            raw_top2_mask, [48, 52, 66, 93, 103]
        )
        raw_mask[top2_verified] = 48

    return thai_dish_info


def finalize_local_mask(
    raw_mask: np.ndarray,
    conf_map: np.ndarray,
    thai_dish_info: Dict[str, object],
) -> Tuple[np.ndarray, float, List[Dict[str, object]]]:
    """Convert 104-class FoodSeg103 mask to 10 local nutrition classes + metadata."""
    h, w = raw_mask.shape[:2]
    local_mask = np.zeros((h, w), dtype=np.uint8)
    detailed_ingredients: List[Dict[str, object]] = []
    unique_ids, counts = np.unique(raw_mask, return_counts=True)

    dish_code = str(thai_dish_info.get("top1_code", ""))
    dish_conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    total_food_px = int(np.sum(counts[unique_ids > 0]))
    min_component_px = min(250, max(20, int(0.001 * max(total_food_px, 1))))
    for cid_np, px_np in sorted(zip(unique_ids, counts), key=lambda x: x[1], reverse=True):
        cid = int(cid_np)
        px = int(px_np)
        if cid == 0 or px < min_component_px:
            continue
        local_cid = FOODSEG103_TO_LOCAL_10CLASS.get(cid, 0)
        if cid in (65, 66) and dish_code in _KHAOPAD_CODES:
            local_cid = 9
        elif (
            dish_code in ("KaiJeowMooSaap", "KaiThoon", "omelette", "SonInLawEggs")
            and dish_conf >= 60.0
            and cid != 66
            and local_cid in (2, 9)
        ):
            local_cid = 4
        elif (
            cid == 56
            and dish_code in ("PadThai", "pad_thai")
            and dish_conf >= 60.0
            and (px / max(total_food_px, 1)) > 0.30
        ):
            local_cid = 9
        if local_cid > 0:
            local_mask[raw_mask == cid] = local_cid
        eng_name = FOODSEG103_ID2LABEL.get(cid, f"class_{cid}")
        thai_name = FOODSEG103_THAI_NAMES.get(cid, eng_name)
        eng_name, thai_name = _apply_thai_culinary_context(
            cid, eng_name, thai_name, thai_dish_info
        )
        detailed_ingredients.append({
            "foodseg_id": cid,
            "class_name": eng_name,
            "thai_name": thai_name,
            "pixels": px,
            "area_pct": round((px / max(total_food_px, 1)) * 100.0, 2),
            "mapped_local_id": local_cid,
        })

    food_pixels = local_mask > 0
    mean_conf = (
        float(np.clip(conf_map[food_pixels].mean(), 0.0, 1.0))
        if np.any(food_pixels)
        else float(np.clip(conf_map.mean(), 0.0, 1.0))
    )
    return local_mask, mean_conf, detailed_ingredients
