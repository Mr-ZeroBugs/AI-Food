/** Take a photo URI from the camera or library, keep a copy, send it to the server, record the meal. */
import { Directory, File, Paths } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { Platform } from 'react-native';

import { IS_CONFIGURED_BUILD } from '@/lib/config';
import { getState, upsertMeal } from '@/lib/store';
import type { AnalyzeResult, Meal } from '@/lib/types';

const TIMEOUT_MS = 45_000;
// The model sees 512 px; 1280 px keeps the saved photo sharp in the app while cutting a 12 MP phone
// photo (3-5 MB) to ~250 KB, which matters on mobile data and for upload time.
const MAX_SIDE = 1280;

function newId() {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
}

async function shrink(uri: string, width?: number, height?: number): Promise<string> {
  try {
    const ctx = ImageManipulator.manipulate(uri);
    if (width && height && Math.max(width, height) > MAX_SIDE) {
      ctx.resize(width >= height ? { width: MAX_SIDE, height: null } : { width: null, height: MAX_SIDE });
    }
    const img = await ctx.renderAsync();
    const out = await img.saveAsync({ format: SaveFormat.JPEG, compress: 0.82 });
    return Platform.OS === 'web' ? await toDataUrl(out.uri) : out.uri;
  } catch {
    return Platform.OS === 'web' ? toDataUrl(uri).catch(() => uri) : uri; // send the original rather than fail
  }
}

/** Web only: blob: URLs die with the page; a data URL survives a reload in storage. */
async function toDataUrl(uri: string): Promise<string> {
  if (uri.startsWith('data:')) return uri;
  const blob = await (await fetch(uri)).blob();
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

/** The picker's and manipulator's files live in the cache and can be purged; meals keep their own copy. */
function keepPhoto(uri: string, id: string): string {
  if (Platform.OS === 'web') return uri;
  try {
    const dir = new Directory(Paths.document, 'meals');
    if (!dir.exists) dir.create({ intermediates: true });
    const dest = new File(dir, `${id}.jpg`);
    new File(uri).copy(dest);
    return dest.uri;
  } catch {
    return uri;
  }
}

async function post(photoUri: string): Promise<AnalyzeResult> {
  const { serverUrl, accessCode, energyLevel } = getState().settings;
  const form = new FormData();
  if (Platform.OS === 'web') {
    const blob = await (await fetch(photoUri)).blob();
    form.append('image', blob, 'meal.jpg');
  } else {
    // React Native's FormData streams the file from its URI.
    form.append('image', { uri: photoUri, name: 'meal.jpg', type: 'image/jpeg' } as unknown as Blob);
  }
  form.append('energy_level', String(energyLevel));

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(`${serverUrl.replace(/\/+$/, '')}/api/analyze-image`, {
      method: 'POST',
      body: form,
      headers: accessCode ? { 'X-Access-Code': accessCode } : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    throw new Error(
      (e as Error).name === 'AbortError'
        ? 'เซิร์ฟเวอร์ตอบช้าเกินไป ลองใหม่อีกครั้ง'
        : IS_CONFIGURED_BUILD
          ? 'ต่ออินเทอร์เน็ตไม่ได้ ตรวจสัญญาณแล้วกดลองอีกครั้ง รูปยังเก็บไว้ในเครื่อง'
          : `ต่อเซิร์ฟเวอร์ไม่ได้ (${serverUrl}) ตรวจที่อยู่ในหน้าตั้งค่าและว่าเครื่องอยู่ Wi-Fi เดียวกัน`,
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = '';
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? '';
    } catch {}
    if (res.status === 401) {
      throw new Error(IS_CONFIGURED_BUILD ? 'แอปเวอร์ชันนี้เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ลองอัปเดตแอป' : 'รหัสเข้าใช้งานไม่ถูกต้อง แก้ได้ในหน้าตั้งค่า');
    }
    if (res.status >= 500) throw new Error('เซิร์ฟเวอร์ขัดข้องชั่วคราว ลองอีกครั้งในอีกสักครู่');
    throw new Error(detail || `เซิร์ฟเวอร์ตอบกลับผิดพลาด (${res.status})`);
  }
  return (await res.json()) as AnalyzeResult;
}

/** Creates the meal immediately (so the result screen can open on it) and fills it in when done. */
export function startAnalysis(picked: { uri: string; width?: number; height?: number }): string {
  const id = newId();
  const draft: Meal = { id, createdAt: Date.now(), photoUri: picked.uri, status: 'analyzing' };
  upsertMeal(draft);
  void (async () => {
    const small = await shrink(picked.uri, picked.width, picked.height);
    await runAnalysis({ ...draft, photoUri: keepPhoto(small, id) });
  })();
  return id;
}

export async function runAnalysis(meal: Meal) {
  upsertMeal({ ...meal, status: 'analyzing', error: undefined });
  try {
    const result = await post(meal.photoUri);
    upsertMeal({ ...meal, status: 'done', result, error: undefined });
  } catch (e) {
    upsertMeal({ ...meal, status: 'error', error: (e as Error).message });
  }
}
