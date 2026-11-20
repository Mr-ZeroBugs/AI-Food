import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.nutrition import build_result, plate_area, servings_vs_target  # noqa: E402


def test_perfect_211_plate_scores_100():
    a = plate_area({"background": 0.4, "vegetable": 0.30, "rice": 0.15, "animal_protein": 0.15})
    assert a["plate_211"]["ratio"] == {"veg": 50.0, "carb": 25.0, "protein": 25.0}
    assert a["plate_211"]["score"] == 100


def test_all_rice_plate_scores_low():
    a = plate_area({"rice": 0.5})
    assert a["plate_211"]["ratio"]["carb"] == 100.0
    # deviation 50 + 75 + 25 = 150 -> score 0
    assert a["plate_211"]["score"] == 0


def test_fruit_reported_but_not_in_211_ratio():
    a = plate_area({"vegetable": 0.2, "rice": 0.1, "animal_protein": 0.1, "fruit": 0.2})
    assert a["group_pct"]["g4_fruit"] == 33.3
    assert a["plate_211"]["ratio"] == {"veg": 50.0, "carb": 25.0, "protein": 25.0}


def test_liquid_excluded_from_plate_but_counted_as_food():
    a = plate_area({"liquid": 0.3, "animal_protein": 0.1})
    assert a["group_pct"]["g1_protein"] == 100.0
    assert a["food_coverage"] == 0.4


def test_mixed_dish_uses_vlm_split_when_given():
    default = plate_area({"mixed_dish": 0.4})
    assert default["group_pct"]["g2_carb"] == 65.0
    vlm = plate_area({"mixed_dish": 0.4}, mixed_split={"g1_protein": 1, "g2_carb": 1, "g3_veg": 2, "g4_fruit": 0})
    assert vlm["group_pct"] == {"g1_protein": 25.0, "g2_carb": 25.0, "g3_veg": 50.0, "g4_fruit": 0.0}
    assert vlm["mixed_split_source"] == "vlm"


def test_legacy_v2_class_names_still_map():
    a = plate_area({"leafy_greens": 0.1, "root_veg": 0.1, "fried_rice": 0.0, "meat_cooked": 0.1, "rice": 0.1})
    assert a["plate_211"]["ratio"] == {"veg": 50.0, "carb": 25.0, "protein": 25.0}


def test_servings_status_against_meal_share():
    s = servings_vs_target({"g2_carb": 6, "g3_veg": 0.5, "g1_protein": 3, "g4_fruit": 1.3, "g5_fat": 4},
                           sugar_tsp=1, energy_level=2000)
    by = {r["group"]: r for r in s["groups"]}
    assert by["g2_carb"]["status"] == "high"      # 6 vs 3.3
    assert by["g3_veg"]["status"] == "low"        # 0.5 vs 1.7
    assert by["g1_protein"]["status"] == "ok"     # 3 vs 3
    assert by["g5_fat"]["status"] == "over"       # 4 vs 2
    assert s["sugar"]["status"] == "ok"


def test_non_food_photo_is_rejected():
    r = build_result({"class_areas": {"background": 0.99, "vegetable": 0.005}, "food_prob": 0.1}, None)
    assert r["is_food"] is False


def test_area_only_result_when_vlm_unavailable():
    r = build_result({"class_areas": {"vegetable": 0.3, "rice": 0.15, "animal_protein": 0.15}, "food_prob": 0.97},
                     None, vlm_error="ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY")
    assert r["is_food"] is True
    assert r["amounts"] is None
    assert r["fat"]["source"] == "unavailable"
    assert any("ANTHROPIC_API_KEY" in w for w in r["warnings"])


def test_fusion_with_vlm_reading():
    vlm = {
        "is_food": True, "dish_name_th": "ข้าวกะเพราหมูสับไข่ดาว", "cooking_method_th": "ผัด/ทอด",
        "items": [
            {"name_th": "ข้าวสวย", "group": "g2_carb", "amount": 3},
            {"name_th": "หมูสับ", "group": "g1_protein", "amount": 3},
            {"name_th": "ไข่ดาว", "group": "g1_protein", "amount": 2},
            {"name_th": "ถั่วฝักยาว", "group": "g3_veg", "amount": 0.3},
        ],
        "mixed_dish_split": {"g1_protein": 0, "g2_carb": 1, "g3_veg": 0, "g4_fruit": 0},
        "oil_tsp": 3.5, "fat_note_th": "น้ำมันผัดและทอดไข่", "sugar_tsp": 1, "uncertainty_note_th": "",
    }
    seg = {"class_areas": {"rice": 0.2, "animal_protein": 0.15, "egg": 0.08, "vegetable": 0.01}, "food_prob": 0.99}
    r = build_result(seg, vlm, energy_level=2000)
    by = {g["group"]: g for g in r["amounts"]["groups"]}
    assert by["g1_protein"]["amount"] == 5
    assert by["g5_fat"]["amount"] == 3.5 and by["g5_fat"]["status"] == "over"
    assert r["fat"]["source"] == "vlm"
    assert r["plate"]["plate_211"]["ratio"]["veg"] < 5
    assert any("ผัก" in t for t in r["advice"])


def test_disagreement_between_models_is_warned():
    vlm = {"is_food": False, "items": [], "oil_tsp": 0, "sugar_tsp": 0}
    r = build_result({"class_areas": {"rice": 0.3}, "food_prob": 0.9}, vlm)
    assert r["is_food"] is False
    assert r["warnings"]
