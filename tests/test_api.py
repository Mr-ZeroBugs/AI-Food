import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend import vlm  # noqa: E402

client = TestClient(app_module.app)


def _jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 180, 100)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


SEG = {"model_version": "test", "class_areas": {"rice": 0.2, "vegetable": 0.2, "animal_protein": 0.1}, "food_prob": 0.95}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["energy_levels"] == [1600, 2000, 2400]


def test_analyze_without_key_returns_area_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = client.post("/api/analyze", json={"image_b64": _jpeg_b64(), "seg": SEG})
    body = r.json()
    assert r.status_code == 200
    assert body["is_food"] and body["amounts"] is None
    assert body["plate"]["plate_211"]["score"] is not None


def test_analyze_with_vlm(monkeypatch):
    captured = {}

    def fake(jpeg, seg_hint=None):
        captured["hint"] = seg_hint
        return {"is_food": True, "dish_name_th": "ข้าวไก่ผัดผัก", "cooking_method_th": "ผัด",
                "items": [{"name_th": "ข้าว", "group": "g2_carb", "amount": 2}],
                "mixed_dish_split": {"g1_protein": 0, "g2_carb": 0, "g3_veg": 0, "g4_fruit": 0},
                "oil_tsp": 2, "fat_note_th": "น้ำมันผัด", "sugar_tsp": 0.5, "uncertainty_note_th": ""}, None

    monkeypatch.setattr(vlm, "analyze_image", fake)
    r = client.post("/api/analyze", json={"image_b64": _jpeg_b64(), "seg": SEG, "energy_level": 1600})
    body = r.json()
    assert body["dish_name_th"] == "ข้าวไก่ผัดผัก"
    assert body["amounts"]["energy_level"] == 1600
    assert "rice 20%" in captured["hint"]


def test_rejects_non_jpeg_and_bad_energy():
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    r = client.post("/api/analyze", json={"image_b64": base64.b64encode(png.getvalue()).decode(), "seg": SEG})
    assert r.status_code == 400
    r = client.post("/api/analyze", json={"seg": SEG, "energy_level": 999})
    assert r.status_code == 400


def test_access_code(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "secret")
    r = client.post("/api/analyze", json={"seg": SEG, "use_vlm": False})
    assert r.status_code == 401
    r = client.post("/api/analyze", json={"seg": SEG, "use_vlm": False}, headers={"X-Access-Code": "secret"})
    assert r.status_code == 200


def _jpeg_bytes(color=(200, 180, 100), size=(96, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_analyze_image_runs_server_model():
    r = client.post("/api/analyze-image", files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"energy_level": "2000"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "is_food" in body and body["image_size"] == [96, 64]
    if body["is_food"]:
        assert body["overlay_png_b64"]


def test_analyze_image_rejects_non_image():
    r = client.post("/api/analyze-image", files={"image": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_analyze_image_rejects_bad_energy():
    r = client.post("/api/analyze-image", files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
                    data={"energy_level": "999"})
    assert r.status_code == 400


def test_analyze_image_on_real_meal():
    sample = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "samples",
                          "chicken_broccoli_rice.jpg")
    with open(sample, "rb") as f:
        r = client.post("/api/analyze-image", files={"image": ("m.jpg", f.read(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["is_food"]
    assert body["plate"]["plate_211"]["score"] is not None
    assert base64.b64decode(body["overlay_png_b64"]).startswith(b"\x89PNG")


def test_rate_limit_returns_429(monkeypatch):
    from backend import ratelimit

    monkeypatch.setattr(ratelimit.limiter, "per_minute", 2)
    monkeypatch.setattr(ratelimit.limiter, "_hits", {})
    codes = [client.post("/api/analyze-image", files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
                         headers={"X-Forwarded-For": "203.0.113.9"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
