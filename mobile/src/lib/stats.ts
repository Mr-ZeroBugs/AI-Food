import type { Meal, PlateRatio } from '@/lib/types';

export const DAY_MS = 86_400_000;

export function startOfDay(t: number) {
  const d = new Date(t);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

export function scored(meals: Meal[]) {
  return meals.filter(
    (m) => m.status === 'done' && m.result?.is_food && m.result.plate.plate_211.score !== null,
  ) as (Meal & { result: Extract<NonNullable<Meal['result']>, { is_food: true }> })[];
}

/** Plate shares averaged over meals (each meal weighs the same: the guideline is per plate). */
export function averageRatio(meals: Meal[]): { ratio: PlateRatio; score: number; n: number } | null {
  const s = scored(meals);
  if (!s.length) return null;
  const sum = { veg: 0, carb: 0, protein: 0 };
  let score = 0;
  for (const m of s) {
    const r = m.result.plate.plate_211.ratio;
    sum.veg += r.veg;
    sum.carb += r.carb;
    sum.protein += r.protein;
    score += m.result.plate.plate_211.score ?? 0;
  }
  const n = s.length;
  return { ratio: { veg: sum.veg / n, carb: sum.carb / n, protein: sum.protein / n }, score: score / n, n };
}

const TH_MONTHS = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'];
const TH_DAYS = ['อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์'];

export function dayLabel(t: number, now = Date.now()) {
  const d0 = startOfDay(t);
  const today = startOfDay(now);
  if (d0 === today) return 'วันนี้';
  if (d0 === today - DAY_MS) return 'เมื่อวาน';
  const d = new Date(t);
  return `วัน${TH_DAYS[d.getDay()]} ${d.getDate()} ${TH_MONTHS[d.getMonth()]}`;
}

export function timeLabel(t: number) {
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')} น.`;
}

/** Plain-language verdict for a 2:1:1 score (0-100). */
export function scoreWord(score: number) {
  if (score >= 85) return 'ใกล้ 2:1:1 มาก';
  if (score >= 70) return 'ค่อนข้างสมดุล';
  if (score >= 50) return 'ยังเอียงไปบางหมู่';
  return 'ห่างจาก 2:1:1';
}
