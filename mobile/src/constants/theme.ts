import '@/global.css';

import { Platform } from 'react-native';

/**
 * Ground is a warm glazed-ceramic white (light) and a deep charcoal (dark): the meal photo is the
 * colour on every screen, so the shell stays neutral and lets the food and the five group colours read.
 * Group colours are the same as the web app and the server overlay so a plate looks identical everywhere.
 */
export const Colors = {
  light: {
    ground: '#F4F3EE',
    surface: '#FFFFFF',
    surfaceRaised: '#FBFAF7',
    separator: 'rgba(28, 27, 23, 0.10)',
    text: '#1C1B17',
    textSecondary: '#5E5C55',
    textTertiary: '#8A877E',
    tint: '#1D7F4E',
    onTint: '#FFFFFF',
    danger: '#B3261E',
    warning: '#8A5A00',
    warningBg: 'rgba(240, 165, 0, 0.14)',
    scrim: 'rgba(12, 12, 10, 0.38)',
  },
  dark: {
    ground: '#0F100E',
    surface: '#1A1B18',
    surfaceRaised: '#22231F',
    separator: 'rgba(244, 243, 238, 0.12)',
    text: '#F4F3EE',
    textSecondary: '#B5B2A8',
    textTertiary: '#86837A',
    tint: '#4CC488',
    onTint: '#08130D',
    danger: '#FF8A80',
    warning: '#F2C46B',
    warningBg: 'rgba(240, 165, 0, 0.16)',
    scrim: 'rgba(0, 0, 0, 0.45)',
  },
} as const;

export type Palette = { [K in keyof typeof Colors.light]: string };

export const Groups = {
  veg: { color: '#22A05A', label: 'ผัก', full: 'หมู่ 3 ผัก', target: 50 },
  carb: { color: '#E09A00', label: 'ข้าว-แป้ง', full: 'หมู่ 2 ข้าว-แป้ง', target: 25 },
  protein: { color: '#E5484D', label: 'โปรตีน', full: 'หมู่ 1 โปรตีน', target: 25 },
  fruit: { color: '#8E4EC6', label: 'ผลไม้', full: 'หมู่ 4 ผลไม้', target: 0 },
} as const;

export type GroupKey = keyof typeof Groups;
export type PlateKey = Exclude<GroupKey, 'fruit'>;
export const PLATE_KEYS: PlateKey[] = ['veg', 'carb', 'protein'];

export const Font = Platform.select({
  web: 'var(--font-ui)',
  default: undefined, // native: the platform face (SF / Roboto + their Thai fallbacks)
});

/** Type scale (pt/sp). Body follows the platform default; numbers use tabular figures. */
export const Type = {
  largeTitle: { fontSize: 34, lineHeight: 41, fontWeight: '700' as const, letterSpacing: 0.2 },
  title: { fontSize: 22, lineHeight: 28, fontWeight: '700' as const },
  headline: { fontSize: 17, lineHeight: 24, fontWeight: '600' as const },
  body: { fontSize: 17, lineHeight: 25, fontWeight: '400' as const },
  callout: { fontSize: 15, lineHeight: 22, fontWeight: '400' as const },
  footnote: { fontSize: 13, lineHeight: 18, fontWeight: '400' as const },
  caption: { fontSize: 12, lineHeight: 16, fontWeight: '500' as const },
  figure: { fontSize: 56, lineHeight: 60, fontWeight: '700' as const, fontVariant: ['tabular-nums' as const] },
};

export const Space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32, xxxl: 48 } as const;
export const Radius = { sm: 10, md: 16, lg: 24, xl: 32, pill: 999 } as const;
export const MaxContentWidth = 640;
