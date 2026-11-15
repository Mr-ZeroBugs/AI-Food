"""Real-world FoodSeg103 (103-class) Mask2Former inference engine and 5-food-group mapper."""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

# Prevent broken local torchvision binary mismatch before importing transformers
if sys.modules.get("torchvision") is None:
    sys.modules["torchvision"] = None  # type: ignore[assignment]

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessorPil,
)


MODEL_REPO = "mericberktas/foodseg-models"
MODEL_SUBFOLDER = "foodseg103/mask2former"
THFOOD50_REPO = "thean/swinv2-tiny-patch4-window8-256-finetuned-THFOOD-50"
FOOD101_REPO = "skylord/swin-finetuned-food101"

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

# Standard FoodSeg103 104-class schema (0 = background, 1..103 = ingredients)
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
    """
    Map FoodSeg103 class IDs (0..103) into the 10 local ingredient classes of nutrition_engine:
    0: bg, 1: rice, 2: noodles, 3: meat_cooked, 4: egg_fried, 5: egg_boiled,
    6: leafy_greens, 7: root_veg, 8: fruit, 9: fried_rice
    """
    mapping: Dict[int, int] = {0: 0}

    # Carb / grain staples -> 1 (rice/bread/potato) or 2 (noodles/pasta) or 9 (mixed/fried carb)
    for cid in (58, 59, 62, 66, 70):
        mapping[cid] = 1
    for cid in (64, 65):
        mapping[cid] = 2
    for cid in (1, 2, 3, 5, 6, 7, 10, 60, 61, 63, 67):
        mapping[cid] = 9

    # Protein / meat / egg / legumes / dairy -> 3 (meat_cooked) or 5 (egg)
    for cid in (12, 15, 17, 18, 19, 21, 22, 23, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 68):
        mapping[cid] = 3
    mapping[24] = 5  # egg
    mapping[4] = 3   # chocolate/fats minor fallback
    mapping[8] = 3
    mapping[9] = 3   # cheese butter

    # Leafy vegetables & herbs -> 6 (leafy_greens)
    for cid in (74, 75, 76, 77, 80, 85, 87, 88, 89, 90, 91, 92, 95, 96, 102):
        mapping[cid] = 6

    # Root/colored vegetables & mushrooms -> 7 (root_veg)
    for cid in (69, 71, 72, 73, 78, 79, 81, 82, 83, 84, 86, 93, 94, 97, 98, 99, 100, 101):
        mapping[cid] = 7

    # Fruits -> 8 (fruit)
    for cid in range(25, 46):
        mapping[cid] = 8
    mapping[20] = 8
    mapping[14] = 8

    # Drinks/sauces/catch-all
    for cid in (11, 13, 16, 52, 57, 103):
        mapping[cid] = 0

    return mapping


FOODSEG103_TO_LOCAL_10CLASS: Dict[int, int] = _build_foodseg_to_local_map()


def _unshift_mericberktas_id(pred_id: int) -> int:
    """Decode shifted label indices from mericberktas/foodseg-models back to canonical FoodSeg103 IDs."""
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

_THFOOD_RICE_DISHES = {
    "KaoManGai",
    "KaoMooDang",
    "KhaoMokGai",
    "KhaoMooTodGratiem",
    "KhaoNiewMaMuang",
    "KkaoKlukKaphi",
    "PhatKaphrao",
    "StewedPorkLeg",
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


def _merge_shrimp_body_tail_components(raw_mask: np.ndarray) -> None:
    """Merge FoodSeg103 'sausage' (49) + attached red tail (94) components into 'shrimp' (56)."""
    m49 = raw_mask == 49
    m94 = raw_mask == 94
    if not np.any(m49) or not np.any(m94):
        return

    kernel = np.ones((5, 5), dtype=np.uint8)
    dilated_94 = cv2.dilate(m94.astype(np.uint8), kernel, iterations=1) > 0
    num49, lbl49, stats49, _ = cv2.connectedComponentsWithStats(m49.astype(np.uint8))
    merged_bodies = np.zeros_like(m49, dtype=bool)

    for cid in range(1, num49):
        if int(stats49[cid, cv2.CC_STAT_AREA]) < 50:
            continue
        comp = lbl49 == cid
        if np.any(comp & dilated_94):
            merged_bodies |= comp

    if not np.any(merged_bodies):
        return

    dilated_bodies = cv2.dilate(merged_bodies.astype(np.uint8), kernel, iterations=1) > 0
    num94, lbl94, stats94, _ = cv2.connectedComponentsWithStats(m94.astype(np.uint8))
    merged_tails = np.zeros_like(m94, dtype=bool)
    for cid in range(1, num94):
        if int(stats94[cid, cv2.CC_STAT_AREA]) < 15:
            continue
        comp = lbl94 == cid
        if np.any(comp & dilated_bodies):
            merged_tails |= comp

    raw_mask[merged_bodies | merged_tails] = 56


def _fuse_multimodel_context(
    raw_mask: np.ndarray,
    raw_top2_mask: np.ndarray,
    thai_dish_info: Dict[str, object],
    food101_info: Dict[str, object],
) -> Dict[str, object]:
    """Fuse FoodSeg103 Top-2 queries + THFOOD-50 + Food-101 without blind overwrites."""
    total_fg = int(np.count_nonzero(raw_mask > 0))
    rice_ratio = int(np.count_nonzero(raw_mask == 66)) / max(1, total_fg)

    th_code = str(thai_dish_info.get("top1_code", ""))
    th_conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    f101_code = str(food101_info.get("top1_code", ""))
    f101_conf = float(food101_info.get("top1_conf_pct", 0.0))

    # Disambiguate curved pink shrimp body+tail only when spatially attached
    if (f101_code in ("fried_rice", "paella") and f101_conf >= 45.0) or rice_ratio >= 0.30:
        _merge_shrimp_body_tail_components(raw_mask)

    existing_top3 = list(thai_dish_info.get("top3", []))
    # 1. Fried Rice via learned Food-101 + FoodSeg103 consensus (since THFOOD-50 lacks KhaoPad)
    if (
        f101_code == "fried_rice"
        and f101_conf >= 45.0
        and th_conf < 65.0
        and rice_ratio >= 0.20
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
    # 2. General international/Asian dish promotion when THFOOD-50 is uncertain
    elif (
        th_conf < 55.0
        and f101_conf >= 55.0
        and f101_code in FOOD101_THAI_NAMES
        and (f101_code != "fried_rice" or rice_ratio >= 0.20)
    ):
        thai_name = FOOD101_THAI_NAMES[f101_code]
        promoted = {"code": f101_code, "thai_name": thai_name, "conf_pct": f101_conf, "source": "Food-101"}
        thai_dish_info = {
            "top1_code": f101_code,
            "top1_thai": thai_name,
            "top1_conf_pct": f101_conf,
            "top3": [promoted] + existing_top3[:2],
        }

    # 3. Verify pale poached chicken in KaoManGai using Mask2Former's Top-2 per-pixel queries
    if (
        str(thai_dish_info.get("top1_code", "")) == "KaoManGai"
        and float(thai_dish_info.get("top1_conf_pct", 0.0)) >= 60.0
    ):
        top2_verified = np.isin(raw_mask, [52, 93, 103]) & np.isin(
            raw_top2_mask, [48, 52, 66, 93, 103]
        )
        raw_mask[top2_verified] = 48

    return thai_dish_info


def _refine_red_chili_slices(
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    thai_dish_info: Dict[str, object],
) -> None:
    """Refine small red chili slices only within spicy Thai dishes."""
    dish_code = str(thai_dish_info.get("top1_code", ""))
    conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
    if dish_code not in _THFOOD_SPICY_CHILI_DISHES or conf < 60.0:
        return

    candidate_region = np.isin(raw_mask, [46, 47, 48, 50, 52, 54, 89])
    if not np.any(candidate_region):
        return

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chili_candidates = (
        candidate_region & ((hue <= 7) | (hue >= 170)) & (sat >= 145) & (val >= 65)
    )
    if int(np.count_nonzero(chili_candidates)) < 15:
        return

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        chili_candidates.astype(np.uint8)
    )
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if 15 <= area <= 800:
            raw_mask[labels == comp_id] = 94


class RealFoodSeg103Engine:
    """Tri-AI engine: SwinV2-Tiny (THFOOD-50) + Swin-Base (Food-101) + Mask2Former (FoodSeg103)."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Mask2FormerImageProcessorPil(
            do_resize=True,
            size={"shortest_edge": 512, "longest_edge": 1024},
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_REPO,
            subfolder=MODEL_SUBFOLDER,
        ).to(self.device)
        self.model.eval()

        self.thai_processor = AutoImageProcessor.from_pretrained(THFOOD50_REPO)
        self.thai_model = AutoModelForImageClassification.from_pretrained(THFOOD50_REPO).to(
            self.device
        )
        self.thai_model.eval()

        self.food101_processor = AutoImageProcessor.from_pretrained(FOOD101_REPO)
        self.food101_model = AutoModelForImageClassification.from_pretrained(FOOD101_REPO).to(
            self.device
        )
        self.food101_model.eval()

    @staticmethod
    @torch.no_grad()
    def _run_classifier_topk(
        pil_img: Image.Image,
        processor: AutoImageProcessor,
        model: AutoModelForImageClassification,
        name_map: Dict[str, str],
        device: torch.device,
        k: int = 3,
    ) -> Dict[str, object]:
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_vals, top_idxs = torch.topk(probs, k=min(k, probs.shape[0]))

        top3: List[Dict[str, object]] = []
        id2label = model.config.id2label
        for val, idx in zip(top_vals, top_idxs):
            code = str(id2label[int(idx)])
            top3.append({
                "code": code,
                "thai_name": name_map.get(code, code.replace("_", " ")),
                "conf_pct": round(float(val) * 100.0, 1),
            })

        best = top3[0]
        return {
            "top1_code": best["code"],
            "top1_thai": best["thai_name"],
            "top1_conf_pct": best["conf_pct"],
            "top3": top3,
        }

    @torch.no_grad()
    def classify_thai_dish(self, pil_img: Image.Image) -> Dict[str, object]:
        """Identify the Thai dish class (top-1 and top-3) using SwinV2-Tiny (THFOOD-50)."""
        return self._run_classifier_topk(
            pil_img, self.thai_processor, self.thai_model, THFOOD50_THAI_NAMES, self.device
        )

    @torch.no_grad()
    def classify_food101_dish(self, pil_img: Image.Image) -> Dict[str, object]:
        """Identify global/Asian dish class (top-1 and top-3) using Swin-Base (Food-101)."""
        return self._run_classifier_topk(
            pil_img, self.food101_processor, self.food101_model, FOOD101_THAI_NAMES, self.device
        )

    @torch.no_grad()
    def predict(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, List[Dict[str, object]], Dict[str, object]]:
        """
        Run Tri-AI inference (Mask2Former FoodSeg103 + SwinV2-Tiny THFOOD-50 + Swin-Base Food-101).
        Returns:
          - local_mask (H, W) uint8 in 0..9 for nutrition_engine
          - raw_mask (H, W) uint8 in 0..103 (canonical FoodSeg103 classes)
          - mean_conf (0..1)
          - latency_ms
          - detailed_ingredients list of context-fused FoodSeg103 classes
          - thai_dish_info dict with fused top-1 and top-3 dish predictions
        """
        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = self.model(pixel_values=pixel_values)

        # Compute per-pixel Top-2 class probabilities directly on GPU
        class_probs = F.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]
        mask_probs = outputs.masks_queries_logits.sigmoid()
        sem_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        sem_probs = F.interpolate(sem_probs, size=(h, w), mode="bilinear", align_corners=False)

        top2_vals, top2_preds = torch.topk(sem_probs, k=2, dim=1)
        thai_dish_info = self.classify_thai_dish(pil_img)
        food101_info = self.classify_food101_dish(pil_img)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_pred_top1 = top2_preds[0, 0].cpu().numpy().astype(np.uint8)
        raw_pred_top2 = top2_preds[0, 1].cpu().numpy().astype(np.uint8)
        raw_mask = _UNSHIFT_LUT[raw_pred_top1]
        raw_top2_mask = _UNSHIFT_LUT[raw_pred_top2]
        conf_map = top2_vals[0, 0].cpu().numpy()

        # Fuse THFOOD-50 + Food-101 + Mask2Former Top-2 queries without hardcoded overfitting
        thai_dish_info = _fuse_multimodel_context(
            raw_mask, raw_top2_mask, thai_dish_info, food101_info
        )

        # Refine fine red chili slices only inside spicy Thai dishes (ID 94: pepper / พริกสด)
        _refine_red_chili_slices(image_rgb, raw_mask, thai_dish_info)

        # Map 104 canonical FoodSeg103 classes -> 10 local classes with Tri-Model dish context
        local_mask = np.zeros((h, w), dtype=np.uint8)
        detailed_ingredients: List[Dict[str, object]] = []
        unique_ids, counts = np.unique(raw_mask, return_counts=True)

        dish_code = str(thai_dish_info.get("top1_code", ""))
        dish_conf = float(thai_dish_info.get("top1_conf_pct", 0.0))
        total_food_px = int(np.sum(counts[unique_ids > 0]))
        for cid_np, px_np in sorted(zip(unique_ids, counts), key=lambda x: x[1], reverse=True):
            cid = int(cid_np)
            px = int(px_np)
            if cid == 0 or px < 35:
                continue
            local_cid = FOODSEG103_TO_LOCAL_10CLASS.get(cid, 0)
            if cid == 66 and dish_code in _KHAOPAD_CODES:
                local_cid = 9
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
        return local_mask, raw_mask, mean_conf, latency_ms, detailed_ingredients, thai_dish_info
