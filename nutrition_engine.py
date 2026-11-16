"""Core nutrition mapping, 2D/2.5D portion estimation, and 2:1:1 evaluation engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


CLASS_NAMES: List[str] = [
    "background",     # 0: พื้นหลัง / ขอบจาน
    "rice",           # 1: ข้าวสวย
    "noodles",        # 2: เส้นก๋วยเตี๋ยว
    "meat_cooked",    # 3: เนื้อสัตว์ปรุงสุก (หมู/ไก่)
    "egg_fried",      # 4: ไข่ดาว
    "egg_boiled",     # 5: ไข่ต้ม
    "leafy_greens",   # 6: ผักใบเขียว
    "root_veg",       # 7: ผักหัว / แตงกวา / มะเขือเทศ / แครอท
    "fruit",          # 8: ผลไม้
    "fried_rice",     # 9: ข้าวผัด (เมนูคลุกปนกัน)
]

CLASS_THAI_NAMES: Dict[str, str] = {
    "background": "พื้นหลัง/จาน",
    "rice": "ข้าวสวย",
    "noodles": "เส้นก๋วยเตี๋ยว",
    "meat_cooked": "เนื้อสัตว์ปรุงสุก",
    "egg_fried": "ไข่ดาว",
    "egg_boiled": "ไข่ต้ม",
    "leafy_greens": "ผักใบเขียว",
    "root_veg": "ผักหลากสี/แตงกวา",
    "fruit": "ผลไม้",
    "fried_rice": "ข้าวผัด",
}

FIVE_GROUP_KEYS: List[str] = [
    "protein",
    "carb",
    "mineral_veg",
    "vitamin_veg_fruit",
    "fat",
]

FIVE_GROUP_LABELS: Dict[str, str] = {
    "protein": "หมู่ 1: โปรตีน",
    "carb": "หมู่ 2: คาร์โบไฮเดรต",
    "mineral_veg": "หมู่ 3: เกลือแร่ (ผัก)",
    "vitamin_veg_fruit": "หมู่ 4: วิตามิน (ผัก/ผลไม้)",
    "fat": "หมู่ 5: ไขมัน",
}

# BGR colors for OpenCV visualization (converted to RGB when returning images)
FIVE_GROUP_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "protein": (239, 68, 68),            # Red
    "carb": (245, 158, 11),              # Amber / Gold
    "mineral_veg": (16, 185, 129),       # Emerald
    "vitamin_veg_fruit": (59, 130, 246), # Blue
    "fat": (168, 85, 247),               # Purple
}

CLASS_COLORS_RGB: List[Tuple[int, int, int]] = [
    (30, 41, 59),     # 0 background
    (253, 224, 71),   # 1 rice
    (251, 191, 36),   # 2 noodles
    (220, 38, 38),    # 3 meat_cooked
    (249, 115, 22),   # 4 egg_fried
    (254, 240, 138),  # 5 egg_boiled
    (22, 163, 74),    # 6 leafy_greens
    (16, 185, 129),   # 7 root_veg
    (59, 130, 246),   # 8 fruit
    (217, 119, 6),    # 9 fried_rice
]

# Every ingredient's weights sum to 1.0 across the 5 food groups
FOOD_GROUP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "rice": {"carb": 1.0},
    "noodles": {"carb": 1.0},
    "meat_cooked": {"protein": 0.85, "fat": 0.15},
    "egg_fried": {"protein": 0.75, "fat": 0.25},
    "egg_boiled": {"protein": 1.0},
    "leafy_greens": {"mineral_veg": 0.70, "vitamin_veg_fruit": 0.30},
    "root_veg": {"mineral_veg": 0.65, "vitamin_veg_fruit": 0.35},
    "fruit": {"vitamin_veg_fruit": 1.0},
    "fried_rice": {"carb": 0.70, "protein": 0.15, "fat": 0.15},
}

# Dominant group for class-level confusion matrix / color overlay in 5-group view
CLASS_TO_PRIMARY_GROUP: Dict[int, str] = {
    1: "carb",
    2: "carb",
    3: "protein",
    4: "protein",
    5: "protein",
    6: "mineral_veg",
    7: "mineral_veg",
    8: "vitamin_veg_fruit",
    9: "carb",
}


def _accumulate_groups(
    mask: np.ndarray,
    weights_map: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Aggregate pixel counts or depth-weighted volumes into the 5 food groups."""
    group_totals: Dict[str, float] = {k: 0.0 for k in FIVE_GROUP_KEYS}
    detected_pixels: Dict[str, int] = {}

    for class_id in range(1, len(CLASS_NAMES)):
        class_mask = mask == class_id
        px_count = int(np.count_nonzero(class_mask))
        if px_count == 0:
            continue

        cls_name = CLASS_NAMES[class_id]
        detected_pixels[cls_name] = px_count

        if weights_map is not None:
            measure = float(np.sum(weights_map[class_mask]))
            if measure <= 0.0:
                measure = float(px_count)
        else:
            measure = float(px_count)

        for group_key, weight in FOOD_GROUP_WEIGHTS[cls_name].items():
            group_totals[group_key] += measure * weight

    return group_totals, detected_pixels


def _normalize_to_pct(totals: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive group totals into percentages summing to 100.0."""
    grand_total = sum(totals.values())
    if grand_total <= 0.0:
        return {k: 0.0 for k in FIVE_GROUP_KEYS}
    return {k: round((v / grand_total) * 100.0, 2) for k, v in totals.items()}


# Calibrated on all 891 laboratory-weighed plates of Nutrition5k train split
# using Depth-Anything-V2-Small relative elevation above the 2D background plane.
TRAIN_CALIBRATED_GROUP_DENSITY: Dict[str, float] = {
    "protein": 0.937,
    "carb": 1.001,
    "mineral_veg": 0.806,
    "vitamin_veg_fruit": 1.195,
    "fat": 0.72,
}


def analyze_plate_from_mask(
    mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    apply_density: bool = True,
) -> Dict[str, object]:
    """
    Compute 2D pixel area ratios, optional 2.5D volume ratios, 2:1:1 compliance,
    health score (0-100), and Thai nutritional recommendations from a segmentation mask.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask (H, W), got shape {mask.shape}")

    totals_2d, detected_pixels = _accumulate_groups(mask, weights_map=None)
    ratios_2d = _normalize_to_pct(totals_2d)

    if depth_map is not None:
        if depth_map.shape != mask.shape:
            depth_map = np.asarray(
                Image.fromarray(depth_map.astype(np.float32)).resize(
                    (mask.shape[1], mask.shape[0]),
                    resample=Image.BILINEAR,
                ),
                dtype=np.float32,
            )
        totals_25d, _ = _accumulate_groups(mask, weights_map=np.clip(depth_map, 0.05, None))
        if apply_density:
            for gk in FIVE_GROUP_KEYS:
                totals_25d[gk] *= TRAIN_CALIBRATED_GROUP_DENSITY.get(gk, 1.0)
        ratios_25d = _normalize_to_pct(totals_25d)
    else:
        totals_25d = dict(totals_2d)
        ratios_25d = dict(ratios_2d)

    total_food_pixels = sum(detected_pixels.values())

    # 2.5D Rice-Bed Occlusion Compensation for Thai topped dishes (e.g., ข้าวราดกะเพรา)
    occlusion_note: Optional[str] = None
    if total_food_pixels > 0:
        rice_pct = (detected_pixels.get("rice", 0) / total_food_pixels) * 100.0
        meat_pct = (detected_pixels.get("meat_cooked", 0) / total_food_pixels) * 100.0
        if 2.0 <= rice_pct <= 25.0 and meat_pct >= 35.0:
            # Estimate occluded rice layer underneath the central meat topping
            hidden_rice_vol = totals_25d["protein"] * 0.85
            totals_25d["carb"] += hidden_rice_vol
            ratios_25d = _normalize_to_pct(totals_25d)
            occlusion_note = (
                f"ตรวจพบเมนูราดข้าว (ข้าวโผล่ขอบจาน {rice_pct:.1f}% ใต้กับข้าว) "
                f"ระบบ 2.5D ชดเชยปริมาณข้าวที่ถูกบดบังให้เป็น {ratios_25d['carb']:.1f}% แล้ว"
            )

    # Evaluate 2:1:1 Healthy Plate using active ratios (prefer 2.5D if provided, else 2D)
    eval_ratios = ratios_25d if depth_map is not None else ratios_2d
    veg_raw = eval_ratios["mineral_veg"] + eval_ratios["vitamin_veg_fruit"]
    carb_raw = eval_ratios["carb"]
    protein_raw = eval_ratios["protein"]
    fat_pct = eval_ratios["fat"]

    main_sum = veg_raw + carb_raw + protein_raw
    if main_sum > 0:
        veg_211 = round((veg_raw / main_sum) * 100.0, 2)
        carb_211 = round((carb_raw / main_sum) * 100.0, 2)
        protein_211 = round((protein_raw / main_sum) * 100.0, 2)
    else:
        veg_211 = carb_211 = protein_211 = 0.0

    # Health score (0-100): deviation from 50:25:25 plus penalty for excess hidden fat
    if total_food_pixels == 0:
        health_score = 0.0
        balance_status = "ไม่พบอาหารในภาพ"
        recommendations = ["กรุณาถ่ายภาพจานอาหารให้ชัดเจนและมีแสงสว่างเพียงพอ"]
    else:
        dev = abs(veg_211 - 50.0) + abs(carb_211 - 25.0) + abs(protein_211 - 25.0)
        fat_penalty = max(0.0, (fat_pct - 10.0) * 1.4)
        health_score = round(max(0.0, min(100.0, 100.0 - dev * 0.72 - fat_penalty)), 1)

        recommendations: List[str] = []
        if veg_211 < 35.0:
            recommendations.append(
                f"ผักและผลไม้ (หมู่ 3-4) มีเพียง {veg_211:.1f}% ต่ำกว่าเกณฑ์ 2:1:1 (50%) แนะนำเพิ่มผักใบเขียวหรือผักสดอีก 1-2 ทัพพี"
            )
        if carb_211 > 38.0:
            recommendations.append(
                f"ข้าว-แป้ง (หมู่ 2) สูงถึง {carb_211:.1f}% (เกณฑ์แนะนำ ~25%) ควรลดปริมาณข้าว/เส้นลง หรือเปลี่ยนเป็นข้าวกล้อง"
            )
        if protein_211 < 16.0:
            recommendations.append(
                f"โปรตีน (หมู่ 1) ค่อนข้างน้อย ({protein_211:.1f}%) แนะนำเสริมไข่ต้ม เนื้อปลา อกไก่ หรือเต้าหู้"
            )
        elif protein_211 > 38.0:
            recommendations.append(
                f"โปรตีน (หมู่ 1) สูง ({protein_211:.1f}%) ควรเพิ่มผักเคียงเพื่อให้สมดุลตามสูตร 2:1:1"
            )
        if fat_pct > 12.0:
            recommendations.append(
                f"ตรวจพบไขมันแฝง (หมู่ 5) ประมาณ {fat_pct:.1f}% จากการทอด/ผัด ควรเลี่ยงของทอดหรือสลับเป็นเมนูต้ม/นึ่ง/ย่าง"
            )
        if occlusion_note:
            recommendations.append(occlusion_note)

        if not recommendations:
            balance_status = "สมดุลตามเกณฑ์จานอาหารสุขภาพ 2:1:1"
            recommendations.append(
                "สัดส่วนอาหารในจานสมดุลดีมาก ใกล้เคียงเกณฑ์ 2:1:1 (ผัก 50% : ข้าว-แป้ง 25% : โปรตีน 25%) สามารถเสริมผลไม้หวานน้อย 1 ส่วนหลังมื้ออาหารได้"
            )
        elif health_score >= 75.0:
            balance_status = "ค่อนข้างสมดุล (ปรับเพิ่มเล็กน้อย)"
        else:
            balance_status = "สัดส่วนยังไม่สมดุลตามเกณฑ์ 2:1:1"

    ingredients_breakdown = (
        [
            {
                "class_name": gk,
                "thai_name": _FIVE_GROUP_TABLE_NAMES[gk],
                "pixels": int(round(totals_2d[gk])),
                "area_pct": ratios_2d[gk],
                "vol_25d_pct": ratios_25d[gk],
            }
            for gk in FIVE_GROUP_KEYS
        ]
        if total_food_pixels > 0
        else []
    )

    return {
        "total_food_pixels": total_food_pixels,
        "five_group_ratios_2d": ratios_2d,
        "five_group_ratios_25d": ratios_25d,
        "plate_211": {
            "veg_pct": veg_211,
            "carb_pct": carb_211,
            "protein_pct": protein_211,
            "fat_warning_pct": fat_pct,
            "target": {"veg_pct": 50.0, "carb_pct": 25.0, "protein_pct": 25.0},
        },
        "health_score": health_score,
        "balance_status": balance_status,
        "recommendations": recommendations,
        "ingredients": ingredients_breakdown,
    }


_FIVE_GROUP_TABLE_NAMES: Dict[str, str] = {
    "protein": "หมู่ 1: โปรตีน (เนื้อสัตว์ / ไข่ / ถั่ว)",
    "carb": "หมู่ 2: คาร์โบไฮเดรต (ข้าว / แป้ง / เส้น)",
    "mineral_veg": "หมู่ 3: เกลือแร่ (ใยอาหาร / ผักใบเขียว-ผักหัว)",
    "vitamin_veg_fruit": "หมู่ 4: วิตามิน (ผักหลากสี / ผลไม้)",
    "fat": "หมู่ 5: ไขมัน (ไขมันธรรมชาติ / น้ำมันผัด-ทอด)",
}

_PLATE_211_CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    1: (245, 158, 11),   # ข้าว-แป้ง (25%)
    2: (245, 158, 11),
    9: (245, 158, 11),
    3: (239, 68, 68),    # โปรตีน (25%)
    4: (239, 68, 68),
    5: (239, 68, 68),
    6: (16, 185, 129),   # ผัก/ผลไม้ (50%)
    7: (16, 185, 129),
    8: (16, 185, 129),
}


def render_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    mode: str = "5groups",
    alpha: float = 0.48,
) -> np.ndarray:
    """Create an RGB visualization overlaying 5-food-group, 2:1:1 pillar, or ingredient colors."""
    color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    for class_id in range(1, len(CLASS_NAMES)):
        region = mask == class_id
        if not np.any(region):
            continue
        if mode == "211":
            color_mask[region] = _PLATE_211_CLASS_COLORS_RGB.get(
                class_id, FIVE_GROUP_COLORS_RGB["carb"]
            )
        elif mode == "5groups":
            group_key = CLASS_TO_PRIMARY_GROUP.get(class_id, "carb")
            color_mask[region] = FIVE_GROUP_COLORS_RGB[group_key]
        else:
            color_mask[region] = CLASS_COLORS_RGB[class_id]

    active = mask > 0
    blended = image_rgb.copy()
    blended[active] = (
        (1.0 - alpha) * image_rgb[active].astype(np.float32)
        + alpha * color_mask[active].astype(np.float32)
    ).astype(np.uint8)
    return blended
