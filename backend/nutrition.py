"""Thai 5-food-group plate analysis.

Two independent measurements, never mixed together:

1. Plate area (from our own segmentation model, runs in the browser)
   -> share of the plate covered by หมู่ 1-4, scored against the 2:1:1 healthy plate
      (ผัก 2 : ข้าว-แป้ง 1 : เนื้อสัตว์ 1, i.e. half the plate vegetables).
      2:1:1 is defined by plate area, so this is what we score.

2. Amount (from the vision-language model, optional)
   -> estimated servings in the Thai Food-Based Dietary Guideline units (ธงโภชนาการ):
      ข้าว-แป้ง = ทัพพี, ผัก = ทัพพี, ผลไม้ = ส่วน, เนื้อสัตว์ = ช้อนกินข้าว, น้ำมัน/น้ำตาล = ช้อนชา,
      compared with one meal's share (1/3) of the daily recommendation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from backend.taxonomy import (
    CLASS_NAMES_TH,
    CLASS_TO_GROUP_SPLIT,
    GROUP_KEYS,
    GROUP_LABELS_TH,
    GROUP_SHORT_TH,
    MIXED_CLASSES,
    NON_PLATE_FOOD_CLASSES,
)

PLATE_GROUPS = ["g1_protein", "g2_carb", "g3_veg", "g4_fruit"]

# ธงโภชนาการ (Thai food-based dietary guideline), daily amounts per energy level.
# Sources: กรมอนามัย ธงโภชนาการ (1,600 / 2,000 / 2,400 kcal tiers); oil and sugar use the
# กรมอนามัย "หวาน มัน เค็ม 6:6:1" daily limits (sugar 6 tsp, oil 6 tsp, salt 1 tsp).
DAILY_TARGETS: Dict[int, Dict[str, float]] = {
    1600: {"g2_carb": 8, "g3_veg": 4, "g4_fruit": 3, "g1_protein": 6, "g5_fat": 6, "sugar": 6},
    2000: {"g2_carb": 10, "g3_veg": 5, "g4_fruit": 4, "g1_protein": 9, "g5_fat": 6, "sugar": 6},
    2400: {"g2_carb": 12, "g3_veg": 6, "g4_fruit": 5, "g1_protein": 12, "g5_fat": 6, "sugar": 6},
}
ENERGY_LEVEL_HINT_TH: Dict[int, str] = {
    1600: "เด็ก 6-13 ปี, หญิงวัยทำงาน, ผู้สูงอายุ",
    2000: "วัยรุ่น 14-25 ปี, ชายวัยทำงาน",
    2400: "หญิง-ชายที่ใช้พลังงานมาก เช่น เกษตรกร นักกีฬา",
}
MEALS_PER_DAY = 3

UNITS: Dict[str, Dict[str, str]] = {
    "g1_protein": {"unit": "spoon", "unit_th": "ช้อนกินข้าว"},
    "g2_carb": {"unit": "ladle", "unit_th": "ทัพพี"},
    "g3_veg": {"unit": "ladle", "unit_th": "ทัพพี"},
    "g4_fruit": {"unit": "portion", "unit_th": "ส่วน"},
    "g5_fat": {"unit": "tsp", "unit_th": "ช้อนชา"},
}

PLATE_211_TARGET = {"veg": 50.0, "carb": 25.0, "protein": 25.0}

# Below this fraction of the image covered by food, the photo is treated as "no food".
MIN_FOOD_COVERAGE = 0.02


def _round(x: float, nd: int = 1) -> float:
    return round(float(x) + 0.0, nd)


def plate_area(
    class_areas: Dict[str, float],
    mixed_split: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Convert per-class pixel fractions into plate-area shares of หมู่ 1-4.

    class_areas: {class_name: fraction of the whole image, 0..1}
    mixed_split: optional per-group split for mixed-dish pixels (from the VLM); replaces the
                 default typical split for "mixed_dish"/"fried_rice" when given.
    """
    group_area = {g: 0.0 for g in PLATE_GROUPS}
    food_area = 0.0
    liquid_area = 0.0
    unknown: List[str] = []
    for name, frac in class_areas.items():
        frac = max(0.0, float(frac))
        if name == "background" or frac == 0.0:
            continue
        if name in NON_PLATE_FOOD_CLASSES:
            liquid_area += frac
            continue
        split = CLASS_TO_GROUP_SPLIT.get(name)
        if split is None:
            unknown.append(name)
            continue
        if name in MIXED_CLASSES and mixed_split:
            split = _normalize(mixed_split, PLATE_GROUPS) or split
        food_area += frac
        for g, w in split.items():
            if g in group_area:
                group_area[g] += frac * w

    total = sum(group_area.values())
    pct = {g: (_round(group_area[g] / total * 100.0) if total > 0 else 0.0) for g in PLATE_GROUPS}

    # 2:1:1 counts vegetables against starch and protein. Fruit is recommended separately
    # (eaten after the meal), so it is reported but not part of the ratio.
    main = group_area["g3_veg"] + group_area["g2_carb"] + group_area["g1_protein"]
    if main > 0:
        r211 = {
            "veg": _round(group_area["g3_veg"] / main * 100.0),
            "carb": _round(group_area["g2_carb"] / main * 100.0),
            "protein": _round(group_area["g1_protein"] / main * 100.0),
        }
        deviation = sum(abs(r211[k] - PLATE_211_TARGET[k]) for k in r211)
        # Worst possible deviation is 150 (e.g. 0:100:0), so this maps linearly to 0-100.
        score = _round(max(0.0, 100.0 * (1.0 - deviation / 150.0)), 0)
    else:
        r211 = {"veg": 0.0, "carb": 0.0, "protein": 0.0}
        score = None

    return {
        "food_coverage": _round(food_area + liquid_area, 4),
        "group_pct": pct,
        "plate_211": {"ratio": r211, "target": PLATE_211_TARGET, "score": score},
        "liquid_coverage": _round(liquid_area, 4),
        "unknown_classes": unknown,
        "mixed_split_source": "vlm" if mixed_split else "default",
    }


def _normalize(d: Dict[str, float], keys: List[str]) -> Optional[Dict[str, float]]:
    vals = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in vals.items()}


def servings_vs_target(
    servings: Dict[str, float],
    sugar_tsp: Optional[float],
    energy_level: int = 2000,
) -> Dict[str, object]:
    """Compare one meal's estimated servings with 1/3 of the daily ธงโภชนาการ amount."""
    daily = DAILY_TARGETS.get(energy_level, DAILY_TARGETS[2000])
    rows = []
    for g in GROUP_KEYS:
        meal_target = daily[g] / MEALS_PER_DAY
        amount = max(0.0, float(servings.get(g, 0.0)))
        ratio = amount / meal_target if meal_target > 0 else 0.0
        if g == "g5_fat":
            status = "over" if ratio > 1.15 else "ok"
        elif ratio < 0.6:
            status = "low"
        elif ratio > 1.5:
            status = "high"
        else:
            status = "ok"
        rows.append({
            "group": g,
            "label_th": GROUP_LABELS_TH[g],
            "amount": _round(amount),
            "unit_th": UNITS[g]["unit_th"],
            "meal_target": _round(meal_target),
            "daily_target": daily[g],
            "status": status,
        })
    sugar = None
    if sugar_tsp is not None:
        sugar_target = daily["sugar"] / MEALS_PER_DAY
        sugar = {
            "amount": _round(sugar_tsp),
            "unit_th": "ช้อนชา",
            "meal_target": _round(sugar_target),
            "status": "over" if sugar_tsp > sugar_target * 1.15 else "ok",
        }
    return {"energy_level": energy_level, "energy_hint_th": ENERGY_LEVEL_HINT_TH.get(energy_level, ""),
            "groups": rows, "sugar": sugar}


def advice(area: Dict[str, object], amounts: Optional[Dict[str, object]]) -> List[str]:
    tips: List[str] = []
    # Whole percentages rounded half-up, the same way the apps display them.
    r = {k: float(int(v + 0.5)) for k, v in area["plate_211"]["ratio"].items()}  # type: ignore[index]
    if area["plate_211"]["score"] is not None:  # type: ignore[index]
        if r["veg"] < 35:
            tips.append(f"ผักมีแค่ {r['veg']:.0f}% ของจาน (เป้าหมายครึ่งจาน) ลองเพิ่มผักอีก 1-2 ทัพพี")
        if r["carb"] > 35:
            tips.append(f"ข้าว-แป้งกินพื้นที่ {r['carb']:.0f}% ของจาน (เป้าหมาย 1 ใน 4) ลองลดข้าวลง หรือเปลี่ยนเป็นข้าวกล้อง")
        if r["protein"] < 15:
            tips.append("โปรตีนน้อยไป ลองเพิ่มไข่ ปลา อกไก่ หรือเต้าหู้")
        elif r["protein"] > 35:
            tips.append(f"เนื้อสัตว์ {r['protein']:.0f}% ของจาน มากกว่าเป้า 1 ใน 4 ลองเพิ่มผักแทน")
    if amounts:
        for row in amounts["groups"]:  # type: ignore[index]
            if row["group"] == "g5_fat" and row["status"] == "over":
                tips.append(
                    f"น้ำมันประมาณ {row['amount']} ช้อนชา เกิน 1 ใน 3 ของที่ควรได้ต่อวัน ({row['meal_target']} ช้อนชา) "
                    "ลองเลือกเมนูต้ม นึ่ง ย่าง แทนทอดหรือผัด"
                )
            if row["group"] == "g4_fruit" and row["amount"] == 0:
                tips.append("มื้อนี้ยังไม่มีผลไม้ (หมู่ 4) กินผลไม้หวานน้อยหลังอาหาร 1 ส่วนได้")
        sugar = amounts.get("sugar")  # type: ignore[union-attr]
        if sugar and sugar["status"] == "over":
            tips.append(f"น้ำตาลประมาณ {sugar['amount']} ช้อนชา ค่อนข้างหวาน (ไม่ควรเกิน 6 ช้อนชาต่อวัน)")
    if not tips:
        tips.append("สัดส่วนจานนี้ใกล้เคียง 2:1:1 ดีแล้ว")
    return tips


def build_result(
    seg: Dict[str, object],
    vlm: Optional[Dict[str, object]],
    energy_level: int = 2000,
    vlm_error: Optional[str] = None,
) -> Dict[str, object]:
    """Fuse the browser segmentation summary with the (optional) VLM reading of the dish."""
    class_areas: Dict[str, float] = dict(seg.get("class_areas") or {})  # type: ignore[arg-type]
    food_prob = float(seg.get("food_prob", 1.0))  # type: ignore[arg-type]

    mixed_split = None
    if vlm and vlm.get("is_food"):
        mixed_split = vlm.get("mixed_dish_split")  # type: ignore[assignment]
    area = plate_area(class_areas, mixed_split=mixed_split)

    seg_says_food = food_prob >= 0.5 and area["food_coverage"] >= MIN_FOOD_COVERAGE  # type: ignore[operator]
    warnings: List[str] = []
    if vlm is not None:
        is_food = bool(vlm.get("is_food"))
        if is_food != seg_says_food:
            warnings.append(
                "โมเดลแบ่งพื้นที่กับโมเดลอ่านเมนูเห็นไม่ตรงกันว่าภาพนี้เป็นอาหารหรือไม่ ผลอาจคลาดเคลื่อน"
            )
    else:
        is_food = seg_says_food
        if vlm_error:
            warnings.append(f"ประเมินปริมาณ/ไขมันไม่ได้ ({vlm_error}) แสดงเฉพาะสัดส่วนพื้นที่จาน")

    if not is_food:
        return {
            "is_food": False,
            "message_th": "ไม่พบจานอาหารชัดเจนในภาพ ลองถ่ายจากด้านบนให้เห็นจานเต็มเฟรม",
            "seg": {"food_prob": round(food_prob, 3), "food_coverage": area["food_coverage"]},
            "warnings": warnings,
        }

    amounts = None
    items: List[Dict[str, object]] = []
    if vlm is not None:
        servings = {g: 0.0 for g in GROUP_KEYS}
        for it in vlm.get("items", []):  # type: ignore[union-attr]
            g = it.get("group")
            if g in servings:
                servings[g] += max(0.0, float(it.get("amount", 0.0)))
            items.append({
                "name_th": it.get("name_th", ""),
                "group": g,
                "group_short_th": GROUP_SHORT_TH.get(str(g), ""),
                "amount": _round(float(it.get("amount", 0.0))),
                "unit_th": UNITS.get(str(g), {}).get("unit_th", ""),
            })
        servings["g5_fat"] += max(0.0, float(vlm.get("oil_tsp", 0.0) or 0.0))
        amounts = servings_vs_target(servings, vlm.get("sugar_tsp"), energy_level)  # type: ignore[arg-type]

        # Cross-check: a group the VLM sees in quantity but the segmenter finds nowhere.
        for g in PLATE_GROUPS:
            if servings[g] >= 1.0 and area["group_pct"][g] < 2.0:  # type: ignore[index]
                warnings.append(
                    f"ตรวจพบ{GROUP_SHORT_TH[g]}จากการอ่านเมนู แต่แบ่งพื้นที่ในภาพไม่เจอ (อาจถูกบังหรือปนอยู่ในอาหาร)"
                )

    detected = sorted(
        ({"class": k, "name_th": CLASS_NAMES_TH.get(k, k), "pct_of_image": _round(v * 100.0)}
         for k, v in class_areas.items() if k != "background" and v > 0.002),
        key=lambda x: -x["pct_of_image"],
    )

    return {
        "is_food": True,
        "dish_name_th": (vlm or {}).get("dish_name_th") or None,
        "cooking_method_th": (vlm or {}).get("cooking_method_th") or None,
        "plate": area,
        "detected_classes": detected,
        "items": items,
        "amounts": amounts,
        "fat": {
            "oil_tsp": _round(float(vlm.get("oil_tsp", 0.0) or 0.0)) if vlm else None,
            "source": "vlm" if vlm else "unavailable",
            "note_th": (vlm or {}).get("fat_note_th")
            or "ไขมันมองไม่เห็นจากภาพโดยตรง ต้องอาศัยการอ่านเมนูและวิธีปรุง",
        },
        "advice": advice(area, amounts),
        "warnings": warnings,
        "confidence_note_th": (vlm or {}).get("uncertainty_note_th"),
        "seg": {"food_prob": round(food_prob, 3), "model": seg.get("model_version")},
    }
