/**
 * App state: settings and the meal log, persisted on the device with AsyncStorage.
 * A tiny external store (no library) so every screen reads the same data through useSyncExternalStore.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';
import { useSyncExternalStore } from 'react';
import { Platform } from 'react-native';

import { BUILD_API_KEY, BUILD_API_URL, IS_CONFIGURED_BUILD } from '@/lib/config';
import type { Meal, Settings } from '@/lib/types';

const MEALS_KEY = 'food5.meals.v1';
const SETTINGS_KEY = 'food5.settings.v1';
const MAX_MEALS = 500;

/** During development the API runs on the same computer as the Expo dev server, on port 8000. */
function defaultServerUrl(): string {
  if (IS_CONFIGURED_BUILD) return BUILD_API_URL;
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  const host = Constants.expoConfig?.hostUri?.split(':')[0];
  return host ? `http://${host}:8000` : 'http://localhost:8000';
}

type State = { ready: boolean; meals: Meal[]; settings: Settings };

let state: State = {
  ready: false,
  meals: [],
  settings: { serverUrl: defaultServerUrl(), accessCode: BUILD_API_KEY, energyLevel: 2000 },
};
const listeners = new Set<() => void>();

function emit(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useStore<T>(select: (s: State) => T): T {
  return useSyncExternalStore(subscribe, () => select(state), () => select(state));
}

export const getState = () => state;

async function persistMeals() {
  // An analysis interrupted by closing the app is saved as an error, never as a forever-spinner.
  const saved = state.meals.map((m) =>
    m.status === 'analyzing' ? { ...m, status: 'error' as const, error: 'ปิดแอประหว่างวิเคราะห์' } : m,
  );
  try {
    await AsyncStorage.setItem(MEALS_KEY, JSON.stringify(saved));
  } catch {
    // Storage full or unavailable: the log stays in memory for this session.
  }
}

/** Release builds always use the server they were built for; only the user's own choices persist. */
function mergeSettings(saved: Partial<Settings>): Settings {
  const merged = { ...state.settings, ...saved };
  return IS_CONFIGURED_BUILD ? { ...merged, serverUrl: BUILD_API_URL, accessCode: BUILD_API_KEY } : merged;
}

export async function loadStore() {
  try {
    const [m, s] = await Promise.all([AsyncStorage.getItem(MEALS_KEY), AsyncStorage.getItem(SETTINGS_KEY)]);
    emit({
      ready: true,
      meals: m ? (JSON.parse(m) as Meal[]) : [],
      settings: s ? mergeSettings(JSON.parse(s) as Partial<Settings>) : state.settings,
    });
  } catch {
    emit({ ready: true });
  }
}

export function upsertMeal(meal: Meal) {
  const rest = state.meals.filter((m) => m.id !== meal.id);
  emit({ meals: [meal, ...rest].sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_MEALS) });
  void persistMeals();
}

export function removeMeal(id: string) {
  emit({ meals: state.meals.filter((m) => m.id !== id) });
  void persistMeals();
}

export function clearMeals() {
  emit({ meals: [] });
  void persistMeals();
}

export function updateSettings(patch: Partial<Settings>) {
  emit({ settings: { ...state.settings, ...patch } });
  void AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)).catch(() => {});
}
