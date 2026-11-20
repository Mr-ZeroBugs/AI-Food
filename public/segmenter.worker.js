/* Runs the food segmentation model in the browser (onnxruntime-web, WASM).
   Messages in:  {type:'init'} | {type:'run', id, bitmap}
   Messages out: {type:'progress', loaded, total} | {type:'ready', meta, backend}
                 {type:'result', id, ...} | {type:'error', id?, message} */

const ORT_VERSION = '1.20.1';
importScripts(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`);
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
// Threads need cross-origin isolation (COOP/COEP headers); fall back to 1 thread otherwise.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const MODEL_DIR = new URL('./models/', self.location.href);
let session = null;
let meta = null;

async function fetchWithCache(url) {
  const cacheName = 'food5-models';
  let cache = null;
  try { cache = await caches.open(cacheName); } catch (_) { /* caches unavailable (private mode) */ }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`โหลดโมเดลไม่ได้ (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    self.postMessage({ type: 'progress', loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  if (cache) {
    try { await cache.put(url, new Response(buf)); } catch (_) { /* quota: just skip caching */ }
  }
  return buf;
}

async function init() {
  const metaRes = await fetch(new URL('model.json', MODEL_DIR), { cache: 'no-cache' });
  meta = await metaRes.json();
  // Version the cache key by file name so a new model is downloaded once and old ones ignored.
  const bytes = await fetchWithCache(new URL(meta.file, MODEL_DIR).href);
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  self.postMessage({ type: 'ready', meta, threads: ort.env.wasm.numThreads });
}

function preprocess(bitmap) {
  // Must match training/train_segformer_v3.py preprocess(): longest side -> 512,
  // normalise with ImageNet mean/std, edge-pad bottom/right to 512x512.
  const S = meta.input;
  const scale = S / Math.max(bitmap.width, bitmap.height);
  const nw = Math.max(16, Math.round(bitmap.width * scale));
  const nh = Math.max(16, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(nw, nh);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, nw, nh);
  const px = ctx.getImageData(0, 0, nw, nh).data;
  const out = new Float32Array(3 * S * S);
  const [m0, m1, m2] = meta.mean;
  const [s0, s1, s2] = meta.std;
  const plane = S * S;
  for (let y = 0; y < S; y++) {
    const sy = Math.min(y, nh - 1);
    for (let x = 0; x < S; x++) {
      const sx = Math.min(x, nw - 1);
      const i = (sy * nw + sx) * 4;
      const o = y * S + x;
      out[o] = (px[i] / 255 - m0) / s0;
      out[plane + o] = (px[i + 1] / 255 - m1) / s1;
      out[2 * plane + o] = (px[i + 2] / 255 - m2) / s2;
    }
  }
  return { tensor: new ort.Tensor('float32', out, [1, 3, S, S]), nw, nh };
}

async function run(id, bitmap) {
  const t0 = performance.now();
  const { tensor, nw, nh } = preprocess(bitmap);
  bitmap.close();
  const outputs = await session.run({ pixel_values: tensor });
  const seg = outputs.seg_logits;
  const foodLogit = outputs.food_logit.data[0];
  const [, C, H, W] = seg.dims;
  const stride = meta.output_stride;
  const ch = Math.max(1, Math.round(nh / stride));
  const cw = Math.max(1, Math.round(nw / stride));
  const data = seg.data;
  const plane = H * W;
  const labels = new Uint8Array(ch * cw);
  const counts = new Float64Array(C);
  let confSum = 0;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const p = y * W + x;
      let best = 0, bestV = data[p], sum = 0;
      for (let c = 1; c < C; c++) {
        const v = data[c * plane + p];
        if (v > bestV) { bestV = v; best = c; }
      }
      for (let c = 0; c < C; c++) sum += Math.exp(data[c * plane + p] - bestV);
      confSum += 1 / sum;
      labels[y * cw + x] = best;
      counts[best] += 1;
    }
  }
  const n = ch * cw;
  // Drop specks: a class covering < 0.3% of the image is almost always noise.
  const classAreas = {};
  for (let c = 0; c < C; c++) {
    const frac = counts[c] / n;
    if (c > 0 && frac < 0.003) {
      for (let i = 0; i < n; i++) if (labels[i] === c) labels[i] = 0;
      continue;
    }
    if (frac > 0) classAreas[meta.classes[c]] = Math.round(frac * 1e4) / 1e4;
  }
  self.postMessage({
    type: 'result', id,
    labels, width: cw, height: ch,
    classAreas,
    foodProb: 1 / (1 + Math.exp(-foodLogit)),
    meanConf: confSum / n,
    ms: Math.round(performance.now() - t0),
  }, [labels.buffer]);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') await init();
    else if (msg.type === 'run') await run(msg.id, msg.bitmap);
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
};
