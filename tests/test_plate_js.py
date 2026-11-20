"""The web demo scores plates in the browser (public/lib/plate.js). It must give the same numbers as
backend/nutrition.py, so random class-area mixes are scored by both and compared."""

import json
import os
import random
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.nutrition import advice, plate_area  # noqa: E402

CLASSES = ["background", "rice", "other_starch", "animal_protein", "egg", "plant_protein",
           "vegetable", "fruit", "mixed_dish", "sweets", "liquid"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_matches_python():
    rng = random.Random(3)
    cases = []
    for _ in range(300):
        picked = rng.sample(CLASSES, rng.randint(1, 6))
        cases.append({c: round(rng.random() * 0.4, 4) for c in picked})
    cases += [{"fruit": 0.3}, {"background": 1.0}, {"liquid": 0.5, "rice": 0.1}]
    js = (
        "import('" + os.path.join(ROOT, "public", "lib", "plate.js").replace("\\", "/").replace("C:", "file:///C:")
        + "').then(m => { const cs = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        " console.log(JSON.stringify(cs.map(c => { const p = m.plateArea(c); return [p.ratio, p.score, m.advice(p)]; }))); })"
    )
    out = subprocess.run(["node", "-e", js], input=json.dumps(cases), capture_output=True, text=True, check=True,
                         encoding="utf-8")
    got = json.loads(out.stdout)
    for c, (ratio, score, tips) in zip(cases, got):
        p = plate_area(c)
        assert p["plate_211"]["ratio"] == pytest.approx(ratio, abs=0.051), c
        assert p["plate_211"]["score"] == score, c
        if score is not None:
            assert advice(p, None) == tips, c
