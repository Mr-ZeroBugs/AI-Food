"""Segmentation class taxonomy (v3) and how each class maps onto the Thai 5 food groups.

The Thai Ministry of Public Health's 5 groups (อาหารหลัก 5 หมู่):
  หมู่ 1  โปรตีน        meat, fish, eggs, milk, dried beans / nuts / tofu
  หมู่ 2  คาร์โบไฮเดรต  rice, starch, noodles, bread, sugar, taro, potato
  หมู่ 3  เกลือแร่       vegetables
  หมู่ 4  วิตามิน        fruit
  หมู่ 5  ไขมัน          oils and fats

หมู่ 5 is almost never visible as its own region (it is cooking oil, coconut milk, fatty
cuts), so the segmenter never claims to measure it. Fat is estimated separately from the
dish and cooking method (see backend/vlm.py) and reported in teaspoons.
"""

from __future__ import annotations

from typing import Dict, List, Optional

GROUP_KEYS: List[str] = ["g1_protein", "g2_carb", "g3_veg", "g4_fruit", "g5_fat"]

GROUP_LABELS_TH: Dict[str, str] = {
    "g1_protein": "หมู่ 1 โปรตีน (เนื้อสัตว์ ไข่ นม ถั่ว)",
    "g2_carb": "หมู่ 2 คาร์โบไฮเดรต (ข้าว แป้ง น้ำตาล)",
    "g3_veg": "หมู่ 3 เกลือแร่ (ผัก)",
    "g4_fruit": "หมู่ 4 วิตามิน (ผลไม้)",
    "g5_fat": "หมู่ 5 ไขมัน (น้ำมัน ไขมัน)",
}

GROUP_SHORT_TH: Dict[str, str] = {
    "g1_protein": "โปรตีน",
    "g2_carb": "ข้าว-แป้ง",
    "g3_veg": "ผัก",
    "g4_fruit": "ผลไม้",
    "g5_fat": "ไขมัน",
}

GROUP_COLORS_RGB: Dict[str, List[int]] = {
    "g1_protein": [229, 72, 77],
    "g2_carb": [240, 165, 0],
    "g3_veg": [34, 160, 90],
    "g4_fruit": [142, 78, 198],
    "g5_fat": [120, 120, 120],
}

# v3 classes predicted by the segmenter. Index = class id in the model output.
CLASSES_V3: List[str] = [
    "background",
    "rice",
    "other_starch",      # noodles, pasta, bread, potato, corn, fries
    "animal_protein",    # meat, poultry, fish, seafood
    "egg",
    "plant_protein",     # tofu, dried beans, nuts
    "vegetable",
    "fruit",
    "mixed_dish",        # foods where groups are mixed together (fried rice, pizza, dumplings)
    "sweets",            # desserts, candy, cake, ice cream
    "liquid",            # soup, broth, drinks
]

# Legacy v2 model classes, kept so the old model keeps working until v3 is trained.
CLASSES_V2: List[str] = [
    "background", "rice", "noodles", "meat_cooked", "egg_fried", "egg_boiled",
    "leafy_greens", "root_veg", "fruit", "fried_rice",
]

CLASS_NAMES_TH: Dict[str, str] = {
    "rice": "ข้าว",
    "other_starch": "เส้น/ขนมปัง/มันฝรั่ง",
    "noodles": "เส้น",
    "animal_protein": "เนื้อสัตว์/ปลา/อาหารทะเล",
    "meat_cooked": "เนื้อสัตว์",
    "egg": "ไข่",
    "egg_fried": "ไข่",
    "egg_boiled": "ไข่",
    "plant_protein": "เต้าหู้/ถั่ว",
    "vegetable": "ผัก",
    "leafy_greens": "ผัก",
    "root_veg": "ผัก",
    "fruit": "ผลไม้",
    "mixed_dish": "อาหารคลุก (ข้าวผัด ฯลฯ)",
    "fried_rice": "ข้าวผัด/อาหารคลุก",
    "sweets": "ขนมหวาน",
    "liquid": "น้ำซุป/เครื่องดื่ม",
}

# How a class's visible area is split across the plate groups (หมู่ 1-4 only).
# A pure class goes 100% to one group. Mixed dishes use a typical split; when the
# vision-language model has identified the dish, its own composition replaces this default.
CLASS_TO_GROUP_SPLIT: Dict[str, Dict[str, float]] = {
    "rice": {"g2_carb": 1.0},
    "other_starch": {"g2_carb": 1.0},
    "noodles": {"g2_carb": 1.0},
    "animal_protein": {"g1_protein": 1.0},
    "meat_cooked": {"g1_protein": 1.0},
    "egg": {"g1_protein": 1.0},
    "egg_fried": {"g1_protein": 1.0},
    "egg_boiled": {"g1_protein": 1.0},
    "plant_protein": {"g1_protein": 1.0},
    "vegetable": {"g3_veg": 1.0},
    "leafy_greens": {"g3_veg": 1.0},
    "root_veg": {"g3_veg": 1.0},
    "fruit": {"g4_fruit": 1.0},
    # Typical Thai fried rice / pad thai: mostly starch, some protein, a little vegetable.
    "mixed_dish": {"g2_carb": 0.65, "g1_protein": 0.25, "g3_veg": 0.10},
    "fried_rice": {"g2_carb": 0.65, "g1_protein": 0.25, "g3_veg": 0.10},
    "sweets": {"g2_carb": 1.0},
}

# Classes that are food but are not part of the plate-area ratio (liquids have no
# meaningful "plate area"; they are reported separately).
NON_PLATE_FOOD_CLASSES = {"liquid"}

MIXED_CLASSES = {"mixed_dish", "fried_rice"}


def classes_for(names: Optional[List[str]]) -> List[str]:
    return list(names) if names else list(CLASSES_V3)


# ---------------------------------------------------------------------------
# Training label remaps
# ---------------------------------------------------------------------------

# FoodSeg103 class id -> v3 class id. Unlisted ids (sauce, cheese/butter, "other") -> background.
_FS = {
    "rice": [66],
    "other_starch": [3, 58, 59, 62, 64, 65, 70],
    "animal_protein": [46, 47, 48, 49, 50, 51, 53, 54, 55, 56],
    "egg": [24],
    "plant_protein": [17, 18, 19, 21, 22, 23, 68],
    "vegetable": [69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                  89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102],
    "fruit": [20] + list(range(25, 46)),
    "mixed_dish": [60, 61, 63, 67],
    "sweets": [1, 2, 4, 5, 6, 7, 8, 10],
    "liquid": [11, 12, 13, 14, 15, 16, 57],
}
FOODSEG103_TO_V3: Dict[int, int] = {}
for _name, _ids in _FS.items():
    for _i in _ids:
        FOODSEG103_TO_V3[_i] = CLASSES_V3.index(_name)

# v2 (10-class) pseudo-label masks -> v3 ids. Used for the THFOOD-50 / Nutrition5k rows whose
# masks were produced by the earlier teacher model.
V2_TO_V3: List[int] = [
    0,                                  # background
    CLASSES_V3.index("rice"),
    CLASSES_V3.index("other_starch"),   # noodles
    CLASSES_V3.index("animal_protein"), # meat_cooked
    CLASSES_V3.index("egg"),            # egg_fried
    CLASSES_V3.index("egg"),            # egg_boiled
    CLASSES_V3.index("vegetable"),      # leafy_greens
    CLASSES_V3.index("vegetable"),      # root_veg
    CLASSES_V3.index("fruit"),
    CLASSES_V3.index("mixed_dish"),     # fried_rice
]
