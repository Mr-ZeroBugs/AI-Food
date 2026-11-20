/** Small shared pieces: themed text, inset-grouped surfaces, screen title, meal rows. */
import { Image } from 'expo-image';
import { router } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import type { ReactNode } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View, type StyleProp, type TextStyle, type ViewStyle } from 'react-native';

import { Font, Radius, Space, Type } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { scoreWord, timeLabel } from '@/lib/stats';
import type { Meal } from '@/lib/types';

type Tone = 'text' | 'textSecondary' | 'textTertiary' | 'tint' | 'danger' | 'warning';

export function T({
  children, kind = 'body', tone = 'text', style, numberOfLines,
}: { children: ReactNode; kind?: keyof typeof Type; tone?: Tone; style?: StyleProp<TextStyle>; numberOfLines?: number }) {
  const t = useTheme();
  return (
    <Text numberOfLines={numberOfLines} style={[Type[kind], { color: t[tone], fontFamily: Font }, style]}>
      {children}
    </Text>
  );
}

export function Group({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  const t = useTheme();
  return <View style={[styles.group, { backgroundColor: t.surface }, style]}>{children}</View>;
}

export function Separator() {
  const t = useTheme();
  return <View style={[styles.sep, { backgroundColor: t.separator }]} />;
}

export function ScreenTitle({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <View style={styles.title}>
      <T kind="largeTitle" style={{ fontWeight: '700' }}>{title}</T>
      {subtitle ? <T kind="callout" tone="textSecondary">{subtitle}</T> : null}
    </View>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <T kind="footnote" tone="textSecondary" style={styles.sectionLabel}>{children}</T>;
}

export function MealRow({ meal }: { meal: Meal }) {
  const t = useTheme();
  const r = meal.result;
  let line: ReactNode;
  if (meal.status === 'analyzing') {
    line = (
      <View style={styles.inline}>
        <ActivityIndicator size="small" color={t.textSecondary} />
        <T kind="footnote" tone="textSecondary">กำลังแยกหมู่อาหาร…</T>
      </View>
    );
  } else if (meal.status === 'error') {
    line = <T kind="footnote" tone="danger" numberOfLines={1}>วิเคราะห์ไม่สำเร็จ แตะเพื่อลองใหม่</T>;
  } else if (r && !r.is_food) {
    line = <T kind="footnote" tone="textSecondary">ไม่พบจานอาหารในภาพ</T>;
  } else if (r && r.is_food && r.plate.plate_211.score !== null) {
    const p = r.plate.plate_211.ratio;
    line = (
      <T kind="footnote" tone="textSecondary" style={{ fontVariant: ['tabular-nums'] }}>
        ผัก {Math.round(p.veg)} · แป้ง {Math.round(p.carb)} · โปรตีน {Math.round(p.protein)}
      </T>
    );
  } else {
    line = <T kind="footnote" tone="textSecondary">มีแต่ผลไม้หรือเครื่องดื่ม</T>;
  }
  const score = r?.is_food ? r.plate.plate_211.score : null;

  return (
    <Pressable
      onPress={() => router.push({ pathname: '/meal/[id]', params: { id: meal.id } })}
      style={({ pressed }) => [styles.row, pressed && { backgroundColor: t.separator }]}
      accessibilityRole="button">
        <Image source={{ uri: meal.photoUri }} style={styles.thumb} contentFit="cover" transition={150} />
        <View style={styles.rowText}>
          <T kind="headline">{score !== null && score !== undefined ? scoreWord(score) : timeLabel(meal.createdAt)}</T>
          {score !== null && score !== undefined ? <T kind="footnote" tone="textTertiary">{timeLabel(meal.createdAt)}</T> : null}
          {line}
        </View>
        {score !== null && score !== undefined ? (
          <T kind="title" style={{ fontVariant: ['tabular-nums'] }}>{Math.round(score)}</T>
        ) : null}
        <SymbolView name={{ ios: 'chevron.right', android: 'chevron_right', web: 'chevron_right' }} size={16} tintColor={t.textTertiary} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  group: { borderRadius: Radius.md, overflow: 'hidden' },
  sep: { height: StyleSheet.hairlineWidth, marginLeft: Space.lg },
  title: { gap: 2, paddingTop: Space.sm, paddingBottom: Space.lg },
  sectionLabel: { marginTop: Space.xl, marginBottom: Space.sm, marginLeft: Space.lg },
  inline: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  row: { flexDirection: 'row', alignItems: 'center', gap: Space.md, paddingHorizontal: Space.lg, paddingVertical: Space.md, minHeight: 76 },
  thumb: { width: 56, height: 56, borderRadius: Radius.sm },
  rowText: { flex: 1, gap: 1 },
});
