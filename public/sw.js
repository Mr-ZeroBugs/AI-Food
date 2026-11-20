// Offline support: the app shell and the onnxruntime-web runtime are cached so KinWhy AI opens and
// analyses without a connection after the first visit. The model file itself is cached by the worker.
const SHELL = 'kinwhy-shell-v2';
const RUNTIME = 'kinwhy-runtime-v1';
const ASSETS = ['./', 'index.html', 'styles.css', 'app.js', 'lib/plate.js', 'lib/segment.js', 'lib/store.js',
  'segmenter.worker.js', 'manifest.webmanifest', 'icons/icon-192.png', 'models/model.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => ![SHELL, RUNTIME, 'food5-models'].includes(k)).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.pathname.startsWith('/api/') || url.pathname.endsWith('.onnx')) return;
  const runtime = url.hostname === 'cdn.jsdelivr.net' || url.hostname.endsWith('fonts.googleapis.com') || url.hostname.endsWith('fonts.gstatic.com');
  if (runtime) {
    // Versioned CDN files never change: cache first.
    e.respondWith(caches.open(RUNTIME).then(async (c) => (await c.match(e.request)) || fetch(e.request).then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })));
    return;
  }
  if (url.origin !== location.origin) return;
  // Own files: network first so updates land at once, cache when offline.
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request, { ignoreSearch: true }).then((r) => r || caches.match('index.html'))));
});
