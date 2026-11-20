// Plate-area reading of a segmented photo. A line-for-line port of backend/nutrition.py
// (plate_area + advice) and backend/taxonomy.py, so the web demo needs no server.
// tests/test_plate_js.py checks this file against the Python version.

export const CLASS_TO_GROUP = {
  rice: { carb: 1 }, other_starch: { carb: 1 }, sweets: { carb: 1 },
  animal_protein: { protein: 1 }, egg: { protein: 1 }, plant_protein: { protein: 1 },
  vegetable: { veg: 1 }, fruit: { fruit: 1 },
  mixed_dish: { carb: 0.65, protein: 0.25, veg: 0.10 },
};
const NON_PLATE = new Set(['liquid']);

export const CLASS_NAMES_TH = {
  rice: 'ข้าว', other_starch: 'เส้น ขนมปัง มันฝรั่ง', animal_protein: 'เนื้อสัตว์ ปลา อาหารทะเล',
  egg: 'ไข่', plant_protein: 'เต้าหู้ ถั่ว', vegetable: 'ผัก', fruit: 'ผลไม้',
  mixed_dish: 'อาหารคลุก', sweets: 'ขนมหวาน', liquid: 'น้ำซุป เครื่องดื่ม',
};

export const TARGET = { veg: 50, carb: 25, protein: 25 };
export const MIN_FOOD_COVERAGE = 0.02;
const r1 = (x) => Math.round(x * 10) / 10;

/** classAreas: {className: fraction of the whole image}. */
export function plateArea(classAreas) {
  const g = { protein: 0, carb: 0, veg: 0, fruit: 0 };
  let food = 0, liquid = 0;
  for (const [name, f0] of Object.entries(classAreas)) {
    const f = Math.max(0, Number(f0) || 0);
    if (name === 'background' || f === 0) continue;
    if (NON_PLATE.has(name)) { liquid += f; continue; }
    const split = CLASS_TO_GROUP[name];
    if (!split) continue;
    food += f;
    for (const [k, w] of Object.entries(split)) g[k] += f * w;
  }
  const main = g.veg + g.carb + g.protein;
  let ratio = { veg: 0, carb: 0, protein: 0 }, score = null;
  if (main > 0) {
    ratio = { veg: r1(g.veg / main * 100), carb: r1(g.carb / main * 100), protein: r1(g.protein / main * 100) };
    const d = Math.abs(ratio.veg - 50) + Math.abs(ratio.carb - 25) + Math.abs(ratio.protein - 25);
    score = Math.round(Math.max(0, 100 * (1 - d / 150)));
  }
  const total = g.protein + g.carb + g.veg + g.fruit;
  return {
    foodCoverage: food + liquid,
    liquidCoverage: liquid,
    fruitPct: total > 0 ? r1(g.fruit / total * 100) : 0,
    ratio, score,
  };
}

export function isFood(plate, foodProb) {
  return foodProb >= 0.5 && plate.foodCoverage >= MIN_FOOD_COVERAGE;
}

export function advice(plate) {
  if (plate.score === null) return [];
  const r = Object.fromEntries(Object.entries(plate.ratio).map(([k, v]) => [k, Math.floor(v + 0.5)]));
  const tips = [];
  if (r.veg < 35) tips.push(`ผักมีแค่ ${r.veg}% ของจาน (เป้าหมายครึ่งจาน) ลองเพิ่มผักอีก 1-2 ทัพพี`);
  if (r.carb > 35) tips.push(`ข้าว-แป้งกินพื้นที่ ${r.carb}% ของจาน (เป้าหมาย 1 ใน 4) ลองลดข้าวลง หรือเปลี่ยนเป็นข้าวกล้อง`);
  if (r.protein < 15) tips.push('โปรตีนน้อยไป ลองเพิ่มไข่ ปลา อกไก่ หรือเต้าหู้');
  else if (r.protein > 35) tips.push(`เนื้อสัตว์ ${r.protein}% ของจาน มากกว่าเป้า 1 ใน 4 ลองเพิ่มผักแทน`);
  if (!tips.length) tips.push('สัดส่วนจานนี้ใกล้เคียง 2:1:1 ดีแล้ว');
  return tips;
}

export function scoreWord(s) {
  if (s >= 85) return 'สมดุลดีมาก';
  if (s >= 70) return 'ค่อนข้างสมดุล';
  if (s >= 50) return 'เกือบแล้ว ปรับอีกนิด';
  return 'ยังไม่สมดุล';
}
