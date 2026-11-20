// Meal log in IndexedDB (photos are Blobs, too big for localStorage). Everything stays on this device.
const DB = 'kinwhy', STORE = 'meals', VER = 1;
let dbp = null;
const listeners = new Set();

function db() {
  if (!dbp) {
    dbp = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB, VER);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' }).createIndex('at', 'createdAt');
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbp;
}

function tx(mode, fn) {
  return db().then((d) => new Promise((resolve, reject) => {
    const t = d.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(out && 'result' in out ? out.result : undefined);
    t.onerror = () => reject(t.error);
  }));
}

const emit = () => listeners.forEach((l) => l());
export const onChange = (l) => (listeners.add(l), () => listeners.delete(l));

/** Newest first. */
export async function allMeals() {
  const list = await tx('readonly', (s) => s.getAll());
  return list.sort((a, b) => b.createdAt - a.createdAt);
}
export const getMeal = (id) => tx('readonly', (s) => s.get(id));
export async function putMeal(m) { await tx('readwrite', (s) => s.put(m)); emit(); }
export async function deleteMeal(id) { await tx('readwrite', (s) => s.delete(id)); emit(); }
export async function clearMeals() { await tx('readwrite', (s) => s.clear()); emit(); }

// Small preferences stay in localStorage, guarded: private windows can throw.
export function pref(key, fallback) {
  try { const v = localStorage.getItem(`kinwhy.${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
}
export function setPref(key, value) {
  try { localStorage.setItem(`kinwhy.${key}`, JSON.stringify(value)); } catch { /* ignore */ }
}
