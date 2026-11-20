import { ScrollView, StyleSheet, View } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { PlateChart } from '@/components/plate-chart';
import { ShareRows } from '@/components/share-rows';
import { Group, MealRow, ScreenTitle, SectionLabel, Separator, T } from '@/components/ui';
import { MaxContentWidth, Space } from '@/constants/theme';
import { useNow } from '@/hooks/use-now';
import { useTheme } from '@/hooks/use-theme';
import { useStore } from '@/lib/store';
import { averageRatio, DAY_MS, dayLabel, scoreWord, startOfDay } from '@/lib/stats';
import type { Meal } from '@/lib/types';

const SHORT_DAYS = ['อา', 'จ', 'อ', 'พ', 'พฤ', 'ศ', 'ส'];

export default function History() {
  const t = useTheme();
  const insets = useSafeAreaInsets();
  const meals = useStore((s) => s.meals);
  const now = useNow();
  const today = startOfDay(now);

  const week = Array.from({ length: 7 }, (_, i) => {
    const start = today - (6 - i) * DAY_MS;
    const dayMeals = meals.filter((m) => m.createdAt >= start && m.createdAt < start + DAY_MS);
    return { start, avg: averageRatio(dayMeals) };
  });
  const weekAvg = averageRatio(meals.filter((m) => m.createdAt >= today - 6 * DAY_MS));

  const byDay = new Map<number, Meal[]>();
  for (const m of meals) {
    const d = startOfDay(m.createdAt);
    byDay.set(d, [...(byDay.get(d) ?? []), m]);
  }

  return (
    <ScrollView
      style={{ backgroundColor: t.ground }}
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={[styles.content, { paddingTop: insets.top + Space.lg, paddingBottom: insets.bottom + 120 }]}>
      <ScreenTitle title="ประวัติ" subtitle="7 วันล่าสุด" />

      <Group style={styles.week}>
        <View style={styles.days}>
          {week.map(({ start, avg }) => (
            <View key={start} style={styles.day}>
              {avg ? (
                <PlateChart ratio={avg.ratio} size={38} />
              ) : (
                <Svg width={38} height={38}>
                  <Circle cx={19} cy={19} r={17} fill="none" stroke={t.separator} strokeWidth={1.5} strokeDasharray="3 3" />
                </Svg>
              )}
              <T kind="caption" tone={start === today ? 'tint' : 'textSecondary'} style={start === today ? { fontWeight: '700' } : undefined}>
                {SHORT_DAYS[new Date(start).getDay()]}
              </T>
            </View>
          ))}
        </View>
        {weekAvg ? (
          <>
            <View style={styles.weekHead}>
              <T kind="headline">{scoreWord(weekAvg.score)}</T>
              <T kind="callout" tone="textSecondary" style={{ fontVariant: ['tabular-nums'] }}>
                เฉลี่ย {Math.round(weekAvg.score)} คะแนน จาก {weekAvg.n} มื้อ
              </T>
            </View>
            <ShareRows ratio={weekAvg.ratio} />
          </>
        ) : (
          <T kind="callout" tone="textSecondary">ยังไม่มีมื้อที่วิเคราะห์ได้ในสัปดาห์นี้ ถ่ายมื้อแรกได้ที่แท็บวันนี้</T>
        )}
      </Group>

      {[...byDay.entries()].map(([d, list]) => {
        const avg = averageRatio(list);
        return (
          <View key={d}>
            <SectionLabel>
              {dayLabel(d, now)}
              {avg ? ` · เฉลี่ย ${Math.round(avg.score)}` : ''}
            </SectionLabel>
            <Group>
              {list.map((m, i) => (
                <View key={m.id}>
                  {i > 0 ? <Separator /> : null}
                  <MealRow meal={m} />
                </View>
              ))}
            </Group>
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: Space.lg, width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center' },
  week: { padding: Space.lg, gap: Space.lg },
  days: { flexDirection: 'row', justifyContent: 'space-between' },
  day: { alignItems: 'center', gap: Space.xs },
  weekHead: { gap: 2 },
});
