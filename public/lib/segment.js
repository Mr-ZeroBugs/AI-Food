// Main-thread client for segmenter.worker.js: loads the model once, then segments photos.
import { advice, CLASS_NAMES_TH, isFood, plateArea } from './plate.js';

export const GROUP_RGB = {
  rice: [224, 154, 0], other_starch: [224, 154, 0], sweets: [224, 154, 0],
  animal_protein: [229, 72, 77], egg: [229, 72, 77], plant_protein: [229, 72, 77],
  vegetable: [34, 160, 90], fruit: [142, 78, 198], mixed_dish: [176, 112, 40], liquid: [60, 140, 203],
};

let worker = null, ready = null, meta = null, nextId = 1;
const pending = new Map();
const status = { state: 'idle', loaded: 0, total: 0, threads: 1 };
const statusListeners = new Set();
export const modelStatus = () => ({ ...status });
export const onModelStatus = (l) => (statusListeners.add(l), () => statusListeners.delete(l));
const setStatus = (p) => { Object.assign(status, p); statusListeners.forEach((l) => l(modelStatus())); };

export function loadModel() {
  if (ready) return ready;
  setStatus({ state: 'loading' });
  worker = new Worker(new URL('../segmenter.worker.js', import.meta.url));
  ready = new Promise((resolve, reject) => {
    worker.onmessage = ({ data: m }) => {
      if (m.type === 'progress') setStatus({ loaded: m.loaded, total: m.total });
      else if (m.type === 'ready') { meta = m.meta; setStatus({ state: 'ready', threads: m.threads, version: m.meta.version }); resolve(); }
      else if (m.type === 'result') { pending.get(m.id)?.resolve(m); pending.delete(m.id); }
      else if (m.type === 'error') {
        if (m.id && pending.has(m.id)) { pending.get(m.id).reject(new Error(m.message)); pending.delete(m.id); }
        else { setStatus({ state: 'error', message: m.message }); ready = null; reject(new Error(m.message)); }
      }
    };
    worker.postMessage({ type: 'init' });
  });
  return ready;
}

/** Downscale a photo for storage (longest side 1280, JPEG). Applies EXIF orientation. */
export async function shrinkPhoto(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const s = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const c = new OffscreenCanvas(w, h);
  c.getContext('2d').drawImage(bmp, 0, 0, w, h);
  bmp.close();
  return { blob: await c.convertToBlob({ type: 'image/jpeg', quality: 0.85 }), width: w, height: h };
}

/** Run the model on a stored photo and turn its output into the plate reading. */
export async function analyze(blob) {
  await loadModel();
  const bitmap = await createImageBitmap(blob);
  const id = nextId++;
  const out = await new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'run', id, bitmap }, [bitmap]);
  });
  const plate = plateArea(out.classAreas);
  const food = isFood(plate, out.foodProb);
  const detected = Object.entries(out.classAreas)
    .filter(([k, v]) => k !== 'background' && v > 0.002)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ name: CLASS_NAMES_TH[k] ?? k, pct: Math.round(v * 1000) / 10 }));
  return {
    isFood: food, plate, advice: food ? advice(plate) : [], detected,
    overlay: await overlayBlob(out.labels, out.width, out.height),
    model: meta?.version, ms: out.ms,
  };
}

async function overlayBlob(labels, w, h) {
  const img = new ImageData(w, h);
  for (let i = 0; i < labels.length; i++) {
    const col = GROUP_RGB[meta.classes[labels[i]]];
    if (!col) continue;
    img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2]; img.data[i * 4 + 3] = 255;
  }
  const small = new OffscreenCanvas(w, h);
  small.getContext('2d').putImageData(img, 0, 0);
  const big = new OffscreenCanvas(w * 4, h * 4); // smooth the 4-px stair steps
  const ctx = big.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w * 4, h * 4);
  return big.convertToBlob({ type: 'image/png' });
}
