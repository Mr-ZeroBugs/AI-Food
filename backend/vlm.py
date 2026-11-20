"""Read the dish with a vision-language model (Claude): what it is, how much of each group,
and the parts a pixel segmenter cannot see (cooking oil, coconut milk, sugar in the sauce)."""

from __future__ import annotations

import base64
import json
import os
from typing import Dict, Optional, Tuple

MODEL = os.environ.get("FOOD_VLM_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """คุณเป็นนักโภชนาการที่ประเมินอาหารจากรูปถ่าย ตามหลักอาหารหลัก 5 หมู่ของกระทรวงสาธารณสุข และหน่วยของธงโภชนาการ

ภารกิจ: ดูภาพหนึ่งมื้อ แล้วบอกว่าเป็นอาหารอะไร แต่ละส่วนประกอบอยู่หมู่ไหน และมีปริมาณเท่าไร

หมู่และหน่วยที่ต้องใช้ (ห้ามใช้หน่วยอื่น):
- g1_protein หมู่ 1 เนื้อสัตว์ ปลา อาหารทะเล ไข่ เต้าหู้ ถั่วเมล็ดแห้ง: หน่วย "ช้อนกินข้าว" (เนื้อสุก 1 ช้อนกินข้าว ≈ 15 กรัม, ไข่ 1 ฟอง = 2 ช้อนกินข้าว, เต้าหู้แข็ง 1/4 แผ่น = 1 ช้อนกินข้าว)
- g2_carb หมู่ 2 ข้าว เส้น ขนมปัง เผือก มัน ข้าวโพด: หน่วย "ทัพพี" (ข้าวสวย 1 ทัพพี ≈ 60 กรัม, ขนมปัง 1 แผ่น = 1 ทัพพี, เส้นสุก 1 ทัพพี ≈ 60 กรัม)
- g3_veg หมู่ 3 ผัก: หน่วย "ทัพพี" (ผักสุก 1 ทัพพี ≈ 40-50 กรัม, ผักสด ~2 เท่าของผักสุก) นับผักที่เป็นส่วนประกอบจริง ไม่นับพริก กระเทียม ใบโหระพาที่ใช้แค่ปรุงรส
- g4_fruit หมู่ 4 ผลไม้: หน่วย "ส่วน" (เช่น กล้วยน้ำว้า 1 ผล, ส้ม 1 ผลกลาง, มะม่วงสุก ครึ่งผล, แตงโม 6-8 ชิ้นคำ)
- หมู่ 5 ไขมัน ห้ามใส่ใน items ให้ประเมินเป็น oil_tsp (ช้อนชา) จากวิธีปรุงและส่วนผสม: น้ำมันที่ใช้ผัด/ทอด กะทิ หนังสัตว์ ไขมันในเนื้อ น้ำมันในข้าวมัน (กะทิ 1 ช้อนกินข้าว ≈ 1 ช้อนชาน้ำมัน)

แนวทางประเมิน:
- ประเมินปริมาณจากขนาดภาชนะและวัตถุอ้างอิงในภาพ (จาน ช้อน ส้อม ตะเกียบ มือ) จานข้าวทั่วไปเส้นผ่านศูนย์กลาง ~9-10 นิ้ว
- นับส่วนที่มองไม่เห็นแต่มีแน่นอนในเมนูนั้น เช่น ข้าวใต้กับข้าว ไข่ในข้าวผัด
- ถ้าเป็นอาหารคลุกปนกัน (ข้าวผัด ผัดไทย ผัดซีอิ๊ว ข้าวหมกไก่) ให้ระบุ mixed_dish_split เป็นสัดส่วนพื้นที่/ปริมาตรโดยประมาณของ g1_protein g2_carb g3_veg g4_fruit ในส่วนที่คลุกกัน รวมกันได้ 1 ถ้าไม่มีอาหารคลุกให้ใส่สัดส่วนของเมนูหลักโดยรวม
- sugar_tsp คือน้ำตาลที่เติมในอาหาร/ซอส/ขนม/เครื่องดื่ม (ช้อนชา) ไม่นับน้ำตาลธรรมชาติในผลไม้
- ถ้าภาพไม่ใช่อาหาร (สัตว์ สิ่งของ โต๊ะเปล่า จานเปล่า) ให้ is_food = false และ items ว่าง
- ประมาณเป็นตัวเลขทศนิยม 1 ตำแหน่ง ถ้าไม่แน่ใจให้บอกใน uncertainty_note_th สั้นๆ ไม่ต้องเดาให้ดูแม่นเกินจริง
- ตอบชื่อเป็นภาษาไทย"""

_GROUP_ENUM = ["g1_protein", "g2_carb", "g3_veg", "g4_fruit"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "dish_name_th": {"type": "string"},
        "cooking_method_th": {"type": "string", "description": "เช่น ผัด ทอด ต้ม นึ่ง ย่าง แกงกะทิ ยำ"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name_th": {"type": "string"},
                    "group": {"type": "string", "enum": _GROUP_ENUM},
                    "amount": {"type": "number"},
                },
                "required": ["name_th", "group", "amount"],
                "additionalProperties": False,
            },
        },
        "mixed_dish_split": {
            "type": "object",
            "properties": {g: {"type": "number"} for g in _GROUP_ENUM},
            "required": _GROUP_ENUM,
            "additionalProperties": False,
        },
        "oil_tsp": {"type": "number"},
        "fat_note_th": {"type": "string", "description": "ไขมันมาจากไหน สั้นๆ"},
        "sugar_tsp": {"type": "number"},
        "uncertainty_note_th": {"type": "string"},
    },
    "required": [
        "is_food", "dish_name_th", "cooking_method_th", "items", "mixed_dish_split",
        "oil_tsp", "fat_note_th", "sugar_tsp", "uncertainty_note_th",
    ],
    "additionalProperties": False,
}


def is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _validate(data: Dict[str, object]) -> Dict[str, object]:
    """The schema is enforced by the API; this guards ranges the schema cannot express."""
    items = []
    for it in data.get("items", []) or []:  # type: ignore[union-attr]
        if it.get("group") in _GROUP_ENUM:
            amt = float(it.get("amount", 0) or 0)
            it["amount"] = min(max(amt, 0.0), 30.0)
            items.append(it)
    data["items"] = items
    data["oil_tsp"] = min(max(float(data.get("oil_tsp", 0) or 0), 0.0), 30.0)
    data["sugar_tsp"] = min(max(float(data.get("sugar_tsp", 0) or 0), 0.0), 40.0)
    return data


def analyze_image(jpeg_bytes: bytes, seg_hint: Optional[str] = None) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Return (parsed result, error message). Never raises for API problems."""
    if not is_configured():
        return None, "ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY"
    import anthropic

    client = anthropic.Anthropic(timeout=45.0, max_retries=1)
    user_text = "วิเคราะห์อาหารในภาพนี้ตามหลัก 5 หมู่"
    if seg_hint:
        user_text += f"\nข้อมูลจากโมเดลแบ่งพื้นที่ (อาจผิดได้ ใช้ประกอบเท่านั้น): {seg_hint}"
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(jpeg_bytes).decode("ascii")}},
                    {"type": "text", "text": user_text},
                ],
            }],
        )
    except anthropic.AuthenticationError:
        return None, "API key ไม่ถูกต้อง"
    except anthropic.RateLimitError:
        return None, "เรียกใช้บ่อยเกินไป ลองใหม่อีกครั้ง"
    except anthropic.APIStatusError as e:
        return None, f"บริการวิเคราะห์เมนูขัดข้อง ({e.status_code})"
    except anthropic.APIConnectionError:
        return None, "เชื่อมต่อบริการวิเคราะห์เมนูไม่ได้"

    if response.stop_reason == "refusal":
        return None, "บริการปฏิเสธการวิเคราะห์ภาพนี้"
    if response.stop_reason == "max_tokens":
        return None, "คำตอบยาวเกินกำหนด"
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        return None, "ไม่ได้รับผลวิเคราะห์"
    try:
        return _validate(json.loads(text)), None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "อ่านผลวิเคราะห์ไม่ได้"
