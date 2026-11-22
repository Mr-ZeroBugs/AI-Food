// Sample meal log for showing the app: 3 days of breakfast / lunch / dinner built from the
// bundled sample photos. Results are real v4 model outputs, precomputed into public/demo/history
// (regenerate with the export script if the model changes). Meals carry `demo: true` so they can
// be removed without touching the user's own photos.
import { allMeals, deleteMeal, putMeal } from './store.js';

// [days ago, hour, minute, sample]. Today only gets the meals whose time has already passed.
const PLAN = [
  [2, 7, 40, 'kaomangai_real'], [2, 12, 20, 'padthai_real'], [2, 18, 45, 'fish_corn_broccoli'],
  [1, 8, 5, 'chicken_broccoli_rice'], [1, 12, 10, 'kaprao_real'], [1, 19, 0, 'shrimp_rice_veggies'],
  [0, 7, 55, 'khaopad_goong_real'], [0, 12, 30, 'nutrition5k_plate'], [0, 18, 30, 'kaomangai_real'],
];

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
      id: `demo-${ago}-${h}`, demo: true, createdAt: d.getTime(),
      photo, width: r.width, height: r.height, status: 'done', result: { ...r.result, overlay },
    });
    added++;
  }
  return added;
}

export async function clearDemoHistory() {
  for (const m of await allMeals()) if (m.demo) await deleteMeal(m.id);
}
