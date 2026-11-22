// Sample meal log for showing the app: 3 days of breakfast / lunch / dinner built from the
// bundled sample photos. Results are real v4 model outputs, precomputed into public/demo/history
// (regenerate with the export script if the model changes). Meals carry `demo: true` so they can
// be removed without touching the user's own photos.
import { allMeals, deleteMeal, putMeal } from './store.js';

// [days ago, hour, minute, sample]. Today only gets the meals whose time has already passed.
// Older days have 2 meals and a different mix each, so the week strip goes up and down.
const PLAN = [
  [6, 8, 10, 'khaopad_goong_real'], [6, 18, 30, 'kaprao_real'],
  [5, 12, 15, 'fish_corn_broccoli'], [5, 19, 10, 'padthai_real'],
  [4, 7, 50, 'kaomangai_real'], [4, 12, 40, 'shrimp_rice_veggies'],
  [3, 12, 5, 'nutrition5k_plate'], [3, 18, 50, 'chicken_broccoli_rice'],
  [2, 7, 40, 'kaomangai_real'], [2, 12, 20, 'padthai_real'], [2, 18, 45, 'fish_corn_broccoli'],
  [1, 8, 5, 'chicken_broccoli_rice'], [1, 12, 10, 'kaprao_real'], [1, 19, 0, 'shrimp_rice_veggies'],
  [0, 7, 55, 'khaopad_goong_real'], [0, 12, 30, 'nutrition5k_plate'], [0, 18, 30, 'kaomangai_real'],
];

// Everyone sees a filled week on their first visit. Visitors who only have sample meals get them
// re-dated on each visit so the week stays current; once they log a real meal (or remove the
// samples from About) we stop touching their history.
const OPT_OUT = 'kinwhy-demo-off';
export async function seedDemoForVisitor() {
  try { if (localStorage.getItem(OPT_OUT)) return false; } catch { /* storage blocked: still seed */ }
  const meals = await allMeals();
  // Once someone logs their own meal, the samples step aside so their real week isn't mixed with ours.
  if (meals.some((m) => !m.demo)) {
    if (meals.some((m) => m.demo)) { await clearDemoHistory(); return true; }
    return false;
  }
  // Re-date only when the samples are stale (none dated today), so a normal reload doesn't rebuild them.
  const today = new Date(); today.setHours(0, 0, 0, 0);
  if (meals.length && meals.some((m) => m.createdAt >= today.getTime())) return false;
  await loadDemoHistory();
  return true;
}

const blob = async (url) => (await fetch(url)).blob();

export async function hasDemoHistory() {
  return (await allMeals()).some((m) => m.demo);
}

export async function loadDemoHistory() {
  await clearDemoHistory();
  const results = await (await fetch('demo/history/results.json')).json();
  const now = Date.now();
  let added = 0;
  for (const [ago, h, min, name] of PLAN) {
    const d = new Date(now); d.setDate(d.getDate() - ago); d.setHours(h, min, 0, 0);
    if (d.getTime() > now) continue;
    const r = results[name];
    const [photo, overlay] = await Promise.all([blob(`samples/${name}.jpg`), blob(`demo/history/${name}-overlay.png`)]);
    await putMeal({
      id: `demo-${ago}-${h}-${min}`, demo: true, createdAt: d.getTime(),
      photo, width: r.width, height: r.height, status: 'done', result: { ...r.result, overlay },
    });
    added++;
  }
  return added;
}

export async function clearDemoHistory({ optOut = false } = {}) {
  if (optOut) try { localStorage.setItem(OPT_OUT, '1'); } catch { /* ignore */ }
  for (const m of await allMeals()) if (m.demo) await deleteMeal(m.id);
}

// Called whenever meals change: drop the samples as soon as a real meal exists. Never re-seeds.
export async function dropDemoIfRealMeals() {
  const meals = await allMeals();
  if (meals.some((m) => !m.demo && m.status === 'done') && meals.some((m) => m.demo)) await clearDemoHistory();
}
