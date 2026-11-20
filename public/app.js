// KinWhy AI web demo: hash-routed single page. The model runs in this browser (lib/segment.js);
// meals and photos are kept in IndexedDB on this device (lib/store.js).
import { scoreWord, TARGET } from './lib/plate.js';
import { analyze, loadModel, modelStatus, onModelStatus, shrinkPhoto } from './lib/segment.js';
import { allMeals, clearMeals, deleteMeal, getMeal, onChange, putMeal } from './lib/store.js';

const $view = document.getElementById('view');
const $file = document.getElementById('file-input');
const $toast = document.getElementById('toast');

const GROUPS = {
  veg: { color: 'var(--veg)', label: 'ผัก' },
  carb: { color: 'var(--carb)', label: 'ข้าว-แป้ง' },
  protein: { color: 'var(--prot)', label: 'โปรตีน' },
};
const PLATE_KEYS = ['veg', 'carb', 'protein'];
const TH_MONTHS = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'];
const TH_MON = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'];
const TH_DAYS = ['อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์'];
const TH_D = ['อา', 'จ', 'อ', 'พ', 'พฤ', 'ศ', 'ส'];
const SAMPLES = [
  ['khaopad_goong_real', 'ข้าวผัดกุ้ง'], ['kaomangai_real', 'ข้าวมันไก่'], ['padthai_real', 'ผัดไทย'],
  ['kaprao_real', 'กะเพราหมูสับ'], ['fish_corn_broccoli', 'ปลา ข้าวโพด บรอกโคลี'], ['chicken_broccoli_rice', 'ไก่ บรอกโคลี ข้าว'],
  ['shrimp_rice_veggies', 'กุ้ง ข้าว ผักรวม'], ['nutrition5k_plate', 'ไก่ ผักย่าง'],
];
// Hero sample shown before the first meal: a real v4 result, precomputed (public/demo).
const DEMO = { photo: 'samples/khaopad_goong_real.jpg', mask: 'demo/khaopad_goong_real-mask.png', name: 'ข้าวผัดกุ้ง', score: 79, ratio: { veg: 53.7, carb: 37.1, protein: 9.2 } };
const SLOTS = [['เช้า', 0, 11], ['กลางวัน', 11, 16], ['เย็น', 16, 24]];
let sampleScores = {};
fetch('demo/samples.json').then((r) => r.json()).then((j) => { sampleScores = j; if (document.querySelector('.sample-row')) render(); }).catch(() => {});
const DAY = 86_400_000;

// ---------- helpers ----------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const startOfDay = (t) => { const d = new Date(t); d.setHours(0, 0, 0, 0); return d.getTime(); };
function dayLabel(t) {
  const d0 = startOfDay(t), today = startOfDay(Date.now());
  if (d0 === today) return 'วันนี้';
  if (d0 === today - DAY) return 'เมื่อวาน';
  const d = new Date(t);
  return `วัน${TH_DAYS[d.getDay()]} ${d.getDate()} ${TH_MON[d.getMonth()]}`;
}
const timeLabel = (t) => { const d = new Date(t); return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')} น.`; };
const fullDate = (t) => { const d = new Date(t); return `${d.getDate()} ${TH_MONTHS[d.getMonth()]} ${d.getFullYear() + 543}`; };

let urls = [];
function blobUrl(b) { const u = URL.createObjectURL(b); urls.push(u); return u; }
function releaseUrls() { urls.forEach((u) => URL.revokeObjectURL(u)); urls = []; }

function toast(msg) {
  $toast.textContent = msg;
  $toast.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { $toast.hidden = true; }, 3200);
}

const scored = (meals) => meals.filter((m) => m.status === 'done' && m.result?.isFood && m.result.plate.score !== null);
function average(meals) {
  const s = scored(meals);
  if (!s.length) return null;
  const sum = { veg: 0, carb: 0, protein: 0 };
  let score = 0;
  for (const m of s) { for (const k of PLATE_KEYS) sum[k] += m.result.plate.ratio[k]; score += m.result.plate.score; }
  return { ratio: { veg: sum.veg / s.length, carb: sum.carb / s.length, protein: sum.protein / s.length }, score: score / s.length, n: s.length };
}

// ---------- pieces ----------
function plateSvg(ratio, size = 124) {
  // Real shares clockwise from 12 o'clock; rim ticks mark where 2:1:1 would put each boundary.
  const c = size / 2, rim = c - 2, food = rim - size * 0.07;
  const polar = (r, f) => { const a = f * 2 * Math.PI - Math.PI / 2; return [c + r * Math.cos(a), c + r * Math.sin(a)]; };
  const total = (ratio.veg + ratio.carb + ratio.protein) || 1;
  let acc = 0, wedges = '';
  for (const k of PLATE_KEYS) {
    const from = acc; acc += ratio[k] / total;
    if (acc - from < 0.004) continue;
    let d;
    if (acc - from > 0.9999) d = `M ${c - food} ${c} a ${food} ${food} 0 1 0 ${2 * food} 0 a ${food} ${food} 0 1 0 ${-2 * food} 0`;
    else {
      const [x0, y0] = polar(food, from), [x1, y1] = polar(food, acc);
      d = `M ${c} ${c} L ${x0} ${y0} A ${food} ${food} 0 ${acc - from > 0.5 ? 1 : 0} 1 ${x1} ${y1} Z`;
    }
    wedges += `<path d="${d}" fill="${GROUPS[k].color}" stroke="var(--surface)" stroke-width="1.5"/>`;
  }
  const ticks = [0, 0.5, 0.75].map((f) => {
    const [x0, y0] = polar(food - 3, f), [x1, y1] = polar(rim + 1, f);
    return `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="var(--ink)" stroke-width="${size < 64 ? 1 : 2}" stroke-linecap="round"/>`;
  }).join('');
  const label = `ผัก ${Math.round(ratio.veg)}% ข้าว-แป้ง ${Math.round(ratio.carb)}% โปรตีน ${Math.round(ratio.protein)}% ของจาน`;
  return `<svg class="plate-svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${label}">
    <circle cx="${c}" cy="${c}" r="${rim}" fill="var(--surface)" stroke="var(--line-strong)"/>${wedges}${ticks}</svg>`;
}

function emptyPlate(size) {
  const c = size / 2;
  return `<svg width="${size}" height="${size}" aria-hidden="true"><circle cx="${c}" cy="${c}" r="${c - 2}" fill="none" stroke="var(--line-strong)" stroke-width="1.5" stroke-dasharray="3 3"/></svg>`;
}

function shares(ratio) {
  return `<div class="shares">${PLATE_KEYS.map((k) => {
    const v = Math.max(0, Math.min(100, ratio[k])), t = TARGET[k], gap = Math.round(v - t);
    const note = Math.abs(gap) < 3 ? `ตรงเป้า ${t}%` : `เป้า ${t}% · ${gap > 0 ? 'เกิน' : 'ขาด'} ${Math.abs(gap)}`;
    return `<div><div class="share-top"><span class="dot" style="background:${GROUPS[k].color}"></span><b>${GROUPS[k].label}</b>
      <b class="num">${Math.round(v)}%</b><span class="target num">${note}</span></div>
      <div class="track"><span style="width:${v}%;background:${GROUPS[k].color}"></span><i style="left:${t}%"></i></div></div>`;
  }).join('')}</div>`;
}

function mealRow(m) {
  const r = m.result;
  let h, s;
  const score = r?.isFood ? r.plate.score : null;
  if (m.status === 'analyzing') { h = timeLabel(m.createdAt); s = 'กำลังแยกหมู่อาหาร…'; }
  else if (m.status === 'error') { h = timeLabel(m.createdAt); s = '<span style="color:var(--danger)">วิเคราะห์ไม่สำเร็จ แตะเพื่อลองใหม่</span>'; }
  else if (r && !r.isFood) { h = timeLabel(m.createdAt); s = 'ไม่พบจานอาหารในภาพ'; }
  else if (score === null) { h = timeLabel(m.createdAt); s = 'มีแต่ผลไม้หรือเครื่องดื่ม'; }
  else {
    const p = r.plate.ratio;
    h = scoreWord(score);
    s = `${timeLabel(m.createdAt)} · <span class="num">ผัก ${Math.round(p.veg)} · แป้ง ${Math.round(p.carb)} · โปรตีน ${Math.round(p.protein)}</span>`;
  }
  return `<a class="meal-row" href="#/meal/${esc(m.id)}">
    <img src="${blobUrl(m.photo)}" alt="" loading="lazy">
    <div class="text"><div class="h">${h}</div><div class="s">${s}</div></div>
    ${score !== null && score !== undefined ? `<div class="big">${score}</div>` : ''}
    <svg class="ic chev" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg></a>`;
}

function modelStatusHtml() {
  const s = modelStatus();
  if (s.state === 'ready' || s.state === 'idle') return '';
  if (s.state === 'error') return `<div class="group" style="margin-bottom:12px"><div class="model-status" style="color:var(--danger)">โหลดโมเดลไม่สำเร็จ ${esc(s.message || '')} <button class="btn quiet" data-action="retry-model">ลองใหม่</button></div></div>`;
  const pct = s.total ? Math.round(s.loaded / s.total * 100) : 0;
  return `<div class="group" style="margin-bottom:12px"><div class="model-status">
    <span>${s.total ? `กำลังโหลดโมเดลครั้งแรก <span class="num">${pct}%</span>` : 'กำลังเตรียมโมเดล…'}</span>
    <span class="bar"><span style="width:${pct}%"></span></span></div></div>`;
}

// ---------- views ----------
const LOGO = `<svg width="22" height="22" viewBox="0 0 34 34" aria-hidden="true"><circle cx="17" cy="17" r="16" fill="#fff"/><path d="M17 4 A13 13 0 0 1 17 30 Z" fill="#22A05A"/><path d="M17 30 A13 13 0 0 1 4 17 L17 17 Z" fill="#E09A00"/><path d="M4 17 A13 13 0 0 1 17 4 L17 17 Z" fill="#E5484D"/><path d="M17 3.5V30.5M3.5 17H17" stroke="#fff" stroke-width="1.6"/></svg>`;

function pills(ratio) {
  return `<div class="pills">${PLATE_KEYS.map((k) => `<span class="pill"><i style="background:${GROUPS[k].color}"></i>${GROUPS[k].label} <b class="num">${Math.round(ratio[k])}%</b></span>`).join('')}</div>`;
}

function hero({ photo, mask, caption, score, ratio, href, tall }) {
  const d = new Date();
  return `<section class="today-hero${tall ? ' tall' : ''}">
    <img class="hero-photo" src="${photo}" alt="">
    ${mask ? `<img class="hero-mask" src="${mask}" alt="">` : ''}
    <div class="hero-shade"></div>
    <div class="hero-chrome">
      <span class="brand glass">${LOGO}<b>KinWhy AI</b></span>
      <span class="date-chip glass">${TH_DAYS[d.getDay()]} ${d.getDate()} ${TH_MON[d.getMonth()]}</span>
    </div>
    <${href ? `a href="${href}"` : 'div'} class="hero-card glass dense">
      <div class="hero-card-top">${plateSvg(ratio, 64)}
        <div class="hc-text"><div class="hc-cap">${caption}</div><div class="verdict">${score === null ? 'ยังไม่มีคะแนน' : scoreWord(score)}</div></div>
        <div class="hc-score num">${score === null ? '–' : Math.round(score)}</div></div>
      ${pills(ratio)}
    </${href ? 'a' : 'div'}>
  </section>`;
}

function slotTile(name, meal) {
  if (!meal) {
    return `<label class="slot empty" for="file-input"><span class="slot-plus" aria-hidden="true">+</span><span class="slot-name">${name}</span><span class="slot-sub">ถ่ายมื้อนี้</span></label>`;
  }
  const r = meal.result, score = r?.isFood ? r.plate.score : null;
  const sub = meal.status === 'analyzing' ? 'กำลังวิเคราะห์' : meal.status === 'error' ? 'ไม่สำเร็จ' : score === null ? (r && !r.isFood ? 'ไม่พบอาหาร' : 'ไม่มีคะแนน') : scoreWord(score);
  return `<a class="slot" href="#/meal/${esc(meal.id)}"><img src="${blobUrl(meal.photo)}" alt="">
    ${score !== null && score !== undefined ? `<span class="slot-score glass num">${score}</span>` : ''}
    <span class="slot-label"><span class="slot-name">${name}</span><span class="slot-sub">${sub}</span></span></a>`;
}

function targetCard() {
  return `<div class="group explain">
    ${plateSvg({ veg: 50, carb: 25, protein: 25 }, 104)}
    <div><h3>จาน 2:1:1 คืออะไร</h3>
      <p>แบ่งจานเป็น 4 ส่วน ตามคำแนะนำของกรมอนามัย</p>
      <ul class="legend-list">
        <li><i style="background:var(--veg)"></i><b>ผัก</b> ครึ่งจาน</li>
        <li><i style="background:var(--carb)"></i><b>ข้าว-แป้ง</b> หนึ่งในสี่</li>
        <li><i style="background:var(--prot)"></i><b>โปรตีน</b> หนึ่งในสี่</li>
      </ul>
      <p class="small muted">ผลไม้กินแยกหลังอาหาร จึงไม่นับในจาน · ขีดดำรอบจานในแอปคือตำแหน่งที่ 2:1:1 ควรแบ่ง</p></div>
  </div>`;
}

function sampleRow() {
  return `<div class="sample-row" role="list">${SAMPLES.map(([k, name]) => {
    const sc = sampleScores[k]?.score;
    return `<button class="sample" data-sample="${k}" role="listitem" aria-label="ลองกับ${name}">
      <img src="samples/${k}.jpg" alt="" loading="lazy"><span class="sample-name">${name}</span>
      ${sc !== undefined && sc !== null ? `<span class="sample-score glass num">${Math.round(sc)}</span>` : ''}</button>`;
  }).join('')}</div>`;
}

async function viewToday() {
  const meals = await allMeals();
  const today = startOfDay(Date.now());
  const todays = meals.filter((m) => m.createdAt >= today);
  const avg = average(todays);
  const latest = todays.find((m) => m.status === 'done' && m.result?.isFood) || todays[0];
  const heroHtml = latest
    ? hero({
      photo: blobUrl(latest.photo),
      mask: latest.result?.overlay ? blobUrl(latest.result.overlay) : null,
      caption: avg ? `จานเฉลี่ยวันนี้ · ${avg.n} มื้อ` : 'มื้อล่าสุด',
      score: avg ? avg.score : null, ratio: avg ? avg.ratio : { veg: 0, carb: 0, protein: 0 },
      href: `#/meal/${esc(latest.id)}`,
    })
    : hero({ photo: DEMO.photo, mask: DEMO.mask, caption: `ตัวอย่าง · ${DEMO.name}`, score: DEMO.score, ratio: DEMO.ratio, tall: true });

  const bySlot = SLOTS.map(([name, from, to]) => [name, todays.filter((m) => { const h = new Date(m.createdAt).getHours(); return h >= from && h < to; })]);
  const extra = bySlot.flatMap(([, list]) => list.slice(1));

  return `<div class="today">
    ${heroHtml}
    <div class="today-body">
      <div id="model-slot">${modelStatusHtml()}</div>
      <header class="today-head"><h1>${todays.length ? 'มื้อของวันนี้' : 'วันนี้กินอะไรดี'}</h1>
        <p>${todays.length ? `${todays.length} มื้อ · ${fullDate(Date.now())}` : 'ถ่ายจากด้านบนให้เห็นทั้งจาน แล้วดูว่าใกล้จาน 2:1:1 แค่ไหน'}</p></header>
      <div class="slots">${bySlot.map(([name, list]) => slotTile(name, list[0])).join('')}</div>
      ${extra.length ? `<div class="section-label">มื้ออื่นวันนี้</div><div class="group">${extra.map(mealRow).join('')}</div>` : ''}
      ${avg ? `<div class="section-label">สัดส่วนเฉลี่ยวันนี้</div><div class="group pad">${shares(avg.ratio)}</div>` : ''}
      <div class="section-label">ลองกับรูปตัวอย่าง · โมเดลทำงานในเครื่องนี้</div>
      ${sampleRow()}
      ${todays.length ? '' : targetCard()}
    </div>
  </div>`;
}

async function viewMeal(id) {
  const m = await getMeal(id);
  if (!m) return `<div class="page"><header class="title"><h1>ไม่พบมื้อนี้</h1><p>อาจถูกลบไปแล้ว</p></header><a class="btn" href="#/">กลับหน้าวันนี้</a></div>`;
  const r = m.result;
  const w = Math.min(window.innerWidth, 680);
  const h = Math.round(Math.min(w * (m.height / m.width), window.innerHeight * 0.62));
  const food = r && r.isFood ? r : null;
  const plate = food?.plate;
  let body = '';
  if (m.status === 'analyzing') {
    body = `<div class="state"><div class="spinner" aria-hidden="true"></div><h2>กำลังแยกพื้นที่หมู่อาหาร</h2><p>โมเดลทำงานในเครื่องนี้ รูปไม่ถูกส่งออกไปไหน</p></div>`;
  } else if (m.status === 'error') {
    body = `<div class="state"><h2 style="color:var(--danger)">วิเคราะห์ไม่สำเร็จ</h2><p>${esc(m.error || '')}</p><button class="btn" data-action="reanalyze" style="margin-top:8px">ลองอีกครั้ง</button></div>`;
  } else if (r && !r.isFood) {
    body = `<div class="state"><h2>ไม่พบจานอาหาร</h2><p>ลองถ่ายจากด้านบนให้เห็นจานเต็มเฟรม และมีแสงพอ</p></div>`;
  } else if (plate && plate.score === null) {
    body = `<div class="state"><h2>ไม่มีผัก ข้าว หรือโปรตีนให้เทียบ</h2><p>จานนี้มีแต่ผลไม้หรือเครื่องดื่ม จึงไม่ได้ให้คะแนน 2:1:1</p></div>`;
  } else if (plate) {
    body = `<div class="panel-head">${plateSvg(plate.ratio, 92)}<div><div class="verdict">${scoreWord(plate.score)}</div>
      <div class="small muted">คะแนนจาน 2:1:1 เต็ม 100</div></div><div class="score">${plate.score}</div></div>${shares(plate.ratio)}`;
  }
  return `<div class="hero" style="height:${h}px;--h:${h - 2}px">
      <img src="${blobUrl(m.photo)}" alt="รูปมื้ออาหาร">
      ${food?.overlay ? `<img class="mask" id="mask" src="${blobUrl(food.overlay)}" alt="">` : ''}
      ${m.status === 'analyzing' ? '<div class="scrim"></div><div class="scan"></div>' : ''}
      <div class="hero-top">
        <a class="chip icon glass" href="#/" aria-label="กลับ"><svg class="ic" viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg></a>
        <button class="chip icon glass" data-action="delete" aria-label="ลบมื้อนี้"><svg class="ic" viewBox="0 0 24 24"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/></svg></button>
      </div>
      ${food?.overlay ? `<div class="hero-bottom"><button class="chip glass" data-action="mask" aria-pressed="true">
        <svg class="ic" viewBox="0 0 24 24"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="2.5"/></svg><span>ซ่อนสีหมู่อาหาร</span></button></div>` : ''}
    </div>
    <div class="panel-wrap"><div class="panel glass dense"><div class="when">${dayLabel(m.createdAt)} · ${timeLabel(m.createdAt)}</div>${body}</div></div>
    ${food ? `<div class="below">
      ${food.advice.length ? `<div class="group pad block"><h3>คำแนะนำ</h3><ul class="tips">${food.advice.map((a) => `<li>${esc(a)}</li>`).join('')}</ul></div>` : ''}
      ${plate.liquidCoverage > 0.05 ? `<div class="warn">มีน้ำซุปหรือเครื่องดื่มในภาพ ส่วนนี้ไม่นับในสัดส่วนจาน</div>` : ''}
      <div class="group pad block"><h3>สิ่งที่โมเดลเห็นในภาพ</h3>
        ${food.detected.map((d) => `<div class="kv"><span>${esc(d.name)}</span><span>${d.pct}% ของภาพ</span></div>`).join('')}
        <div class="legend"><span><i style="background:var(--veg)"></i>ผัก</span><span><i style="background:var(--carb)"></i>ข้าว-แป้ง</span><span><i style="background:var(--prot)"></i>โปรตีน</span><span><i style="background:var(--fruit)"></i>ผลไม้ (ไม่นับใน 2:1:1)</span><span><i style="background:#B07028"></i>อาหารคลุก</span><span><i style="background:#3C8CCB"></i>น้ำซุป</span></div>
        <p class="small muted" style="margin:14px 0 0">สัดส่วนคิดจากพื้นที่ที่มองเห็นในภาพ ไม่ใช่น้ำหนัก อาหารที่ซ้อนกันหรือน้ำมันที่ใช้ปรุงจะมองไม่เห็น · โมเดล ${esc(food.model || '-')} · ${food.ms} ms</p>
      </div></div>` : '<div class="below"></div>'}`;
}

async function viewHistory() {
  const meals = await allMeals();
  const today = startOfDay(Date.now());
  const week = Array.from({ length: 7 }, (_, i) => {
    const start = today - (6 - i) * DAY;
    return { start, avg: average(meals.filter((m) => m.createdAt >= start && m.createdAt < start + DAY)) };
  });
  const wavg = average(meals.filter((m) => m.createdAt >= today - 6 * DAY));
  const byDay = new Map();
  for (const m of meals) { const d = startOfDay(m.createdAt); if (!byDay.has(d)) byDay.set(d, []); byDay.get(d).push(m); }
  return `<div class="page">
    <header class="title"><h1>ประวัติ</h1><p>7 วันล่าสุด</p></header>
    <div class="group week">
      <div class="days">${week.map((d) => `<div class="day${d.start === today ? ' today' : ''}">${d.avg ? plateSvg(d.avg.ratio, 38) : emptyPlate(38)}<span>${TH_D[new Date(d.start).getDay()]}</span></div>`).join('')}</div>
      ${wavg ? `<div><div class="verdict" style="font-size:17px">${scoreWord(wavg.score)}</div><div class="callout muted num">เฉลี่ย ${Math.round(wavg.score)} คะแนน จาก ${wavg.n} มื้อ</div></div>${shares(wavg.ratio)}`
        : `<div class="week-empty"><p class="callout muted">ยังไม่มีมื้อในสัปดาห์นี้ พอถ่ายแล้วแต่ละวันจะกลายเป็นจานเล็กๆ ตรงนี้ บอกสัดส่วนของวันนั้น</p>
            <label class="btn" for="file-input">ถ่ายมื้อแรก</label></div>`}
    </div>
    ${[...byDay.entries()].map(([d, list]) => { const a = average(list); return `<div class="section-label">${dayLabel(d)}${a ? ` · เฉลี่ย ${Math.round(a.score)}` : ''}</div><div class="group">${list.map(mealRow).join('')}</div>`; }).join('')}
  </div>`;
}

async function viewAbout() {
  const n = (await allMeals()).length;
  const s = modelStatus();
  return `<div class="page">
    <header class="title"><h1>เกี่ยวกับ</h1><p>KinWhy AI · เดโมบนเว็บ</p></header>
    <div class="group prose">
      <h2>ทำงานอย่างไร</h2>
      <p>โมเดล SegFormer-B2 ที่เทรนเองจะระบายว่าพิกเซลไหนในภาพเป็นข้าว-แป้ง โปรตีน ผัก หรือผลไม้ แล้วนำพื้นที่มาเทียบกับจานสุขภาพ 2:1:1 ของกรมอนามัย</p>
      <p>โมเดลรันในเบราว์เซอร์นี้ทั้งหมด รูปไม่ถูกส่งออกจากเครื่อง ครั้งแรกจะโหลดโมเดลขนาด 55 MB จากนั้นเก็บไว้ในเครื่อง และใช้งานได้แม้ไม่มีอินเทอร์เน็ต</p>
      <p class="small muted">ผลเป็นการประมาณจากพื้นที่ที่มองเห็นในภาพ ใช้ประกอบการกินให้สมดุล ไม่ใช่คำแนะนำทางการแพทย์ · ทดสอบบนอาหารไทย 149 จาน คลาดเคลื่อนเฉลี่ยราว 12 จุดเปอร์เซ็นต์</p>
    </div>
    <div class="section-label">ในเครื่องนี้</div>
    <div class="group">
      <div class="row-link"><span>โมเดล</span><span class="end num">${s.state === 'ready' ? `${esc(s.version)} · ${s.threads} เธรด` : s.state === 'loading' ? 'กำลังโหลด' : 'ยังไม่ได้โหลด'}</span></div>
      <button class="row-link" data-action="clear" ${n ? '' : 'disabled style="opacity:.4"'}><span style="color:var(--danger);font-weight:600">ลบประวัติทั้งหมด</span><span class="end num">${n} มื้อ</span></button>
    </div>
    <div class="section-label">จาน 2:1:1</div>
    ${targetCard()}
    <div class="section-label">ข้อมูล</div>
    <div class="group">
      <a class="row-link" href="privacy.html"><span>นโยบายความเป็นส่วนตัว</span><span class="end">›</span></a>
    </div>
  </div>`;
}

// ---------- capture & analysis ----------
async function startFromBlob(fileOrBlob) {
  let small;
  try { small = await shrinkPhoto(fileOrBlob); } catch { toast('เปิดรูปนี้ไม่ได้ ลองรูปอื่น'); return; }
  const meal = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`, createdAt: Date.now(), photo: small.blob, width: small.width, height: small.height, status: 'analyzing' };
  await putMeal(meal);
  location.hash = `#/meal/${meal.id}`;
  runAnalysis(meal);
}

async function runAnalysis(meal) {
  await putMeal({ ...meal, status: 'analyzing', error: undefined });
  try {
    const result = await analyze(meal.photo);
    await putMeal({ ...meal, status: 'done', result, error: undefined });
  } catch (e) {
    await putMeal({ ...meal, status: 'error', error: `โมเดลทำงานไม่สำเร็จ (${e.message || e}) ลองรีเฟรชหน้าแล้วลองใหม่` });
  }
}

$file.addEventListener('change', () => {
  const f = $file.files?.[0];
  $file.value = '';
  if (f) startFromBlob(f);
});

$view.addEventListener('click', async (e) => {
  const sample = e.target.closest('[data-sample]');
  if (sample) {
    const res = await fetch(`samples/${sample.dataset.sample}.jpg`);
    return startFromBlob(await res.blob());
  }
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const id = currentMealId();
  switch (btn.dataset.action) {
    case 'mask': {
      const mask = document.getElementById('mask');
      const on = mask.classList.toggle('off') === false;
      btn.setAttribute('aria-pressed', String(on));
      btn.querySelector('span').textContent = on ? 'ซ่อนสีหมู่อาหาร' : 'แสดงสีหมู่อาหาร';
      break;
    }
    case 'delete':
      if (confirm('ลบมื้อนี้ออกจากประวัติ? รูปและผลวิเคราะห์จะถูกลบออกจากเครื่องนี้')) { await deleteMeal(id); location.hash = '#/'; }
      break;
    case 'reanalyze': { const m = await getMeal(id); if (m) runAnalysis(m); break; }
    case 'clear':
      if (confirm('ลบประวัติทั้งหมด? รูปและผลวิเคราะห์ในเครื่องนี้จะหายทั้งหมด กู้คืนไม่ได้')) { await clearMeals(); toast('ลบประวัติแล้ว'); }
      break;
    case 'retry-model': loadModel().catch(() => {}); render(); break;
  }
});

// ---------- router ----------
const currentMealId = () => (location.hash.match(/^#\/meal\/(.+)$/) || [])[1];
let renderSeq = 0;
async function render() {
  const seq = ++renderSeq;
  const hash = location.hash || '#/';
  const tab = hash.startsWith('#/history') ? 'history' : hash.startsWith('#/about') ? 'about' : hash.startsWith('#/meal') ? null : 'today';
  const prevUrls = urls; urls = [];
  let html;
  if (hash.startsWith('#/meal/')) html = await viewMeal(decodeURIComponent(currentMealId()));
  else if (tab === 'history') html = await viewHistory();
  else if (tab === 'about') html = await viewAbout();
  else html = await viewToday();
  if (seq !== renderSeq) { releaseUrls(); urls = prevUrls; return; }
  const sameRoute = render.last === hash;
  const y = window.scrollY;
  $view.innerHTML = html;
  // The result panel rises once when the meal opens, not again each time its data updates.
  if (sameRoute) $view.querySelectorAll('.panel, .hero-photo, .hero-mask, .hero-card').forEach((p) => { p.style.animation = 'none'; p.style.clipPath = 'none'; });
  prevUrls.forEach((u) => URL.revokeObjectURL(u));
  document.querySelectorAll('.tab').forEach((a) => {
    if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  document.title = tab === 'history' ? 'ประวัติ · KinWhy AI' : tab === 'about' ? 'เกี่ยวกับ · KinWhy AI' : tab ? 'KinWhy AI' : 'ผลของมื้อ · KinWhy AI';
  if (sameRoute) window.scrollTo(0, y); else { window.scrollTo(0, 0); $view.focus({ preventScroll: true }); }
  render.last = hash;
}
window.addEventListener('hashchange', render);
onChange(render);
onModelStatus(() => {
  const slot = document.getElementById('model-slot');
  if (slot) slot.innerHTML = modelStatusHtml();
});

// ---------- liquid-glass refraction map (Chromium) ----------
function buildRefraction() {
  const bar = document.querySelector('.glass-refract');
  if (!bar || !('userAgentData' in navigator)) return; // Safari/Firefox ignore url() in backdrop-filter
  const { width: w, height: h } = bar.getBoundingClientRect();
  const W = Math.round(w), H = Math.round(h), R = H / 2, edge = 16;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d'); const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    // Signed distance to the pill's border and the direction away from its centre line.
    const cx = Math.min(Math.max(x, R), W - R), dx = x - cx, dy = y - R;
    const dist = R - Math.hypot(dx, dy);
    const k = Math.max(0, 1 - dist / edge) ** 2; // 0 inside, 1 at the rim
    const len = Math.hypot(dx, dy) || 1;
    const i = (y * W + x) * 4;
    img.data[i] = 128 - (dx / len) * k * 127; img.data[i + 1] = 128 - (dy / len) * k * 127; img.data[i + 2] = 128; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const fe = document.getElementById('lg-map');
  fe.setAttribute('href', c.toDataURL());
  fe.setAttribute('width', W); fe.setAttribute('height', H);
  document.documentElement.classList.add('has-refract');
}

// ---------- boot ----------
render();
requestAnimationFrame(buildRefraction);
window.addEventListener('resize', () => { clearTimeout(buildRefraction.t); buildRefraction.t = setTimeout(buildRefraction, 200); });
// Start the 55 MB model download in the background so the first photo is quick.
loadModel().catch(() => {});
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
