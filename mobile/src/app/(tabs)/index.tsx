import { ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { CAPTURE_BAR_SPACE, CaptureBar } from '@/components/capture-bar';
import { PlateChart } from '@/components/plate-chart';
import { ShareRows } from '@/components/share-rows';
import { Group, MealRow, ScreenTitle, SectionLabel, Separator, T } from '@/components/ui';
import { MaxContentWidth, Space } from '@/constants/theme';
import { useNow } from '@/hooks/use-now';
import { useTheme } from '@/hooks/use-theme';
import { useStore } from '@/lib/store';
import { averageRatio, dayLabel, scoreWord, startOfDay } from '@/lib/stats';

const TH_MONTHS = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'];

export default function Today() {
  const t = useTheme();
  const insets = useSafeAreaInsets();
  const meals = useStore((s) => s.meals);
  const now = useNow();
  const today = startOfDay(now);
  const todays = meals.filter((m) => m.createdAt >= today);
  const avg = averageRatio(todays);
  const d = new Date(now);

  return (
    <View style={{ flex: 1, backgroundColor: t.ground }}>
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerStyle={[styles.content, { paddingTop: insets.top + Space.lg, paddingBottom: CAPTURE_BAR_SPACE + insets.bottom }]}>
        <ScreenTitle title={dayLabel(now, now)} subtitle={`${d.getDate()} ${TH_MONTHS[d.getMonth()]} ${d.getFullYear() + 543}`} />

        {avg ? (
          <Group style={styles.summary}>
            <View style={styles.summaryTop}>
              <PlateChart ratio={avg.ratio} size={124} />
              <View style={styles.summaryText}>
                <T kind="footnote" tone="textSecondary">จานเฉลี่ยของวันนี้ ({avg.n} มื้อ)</T>
                <T kind="figure">{Math.round(avg.score)}</T>
                <T kind="callout" tone="textSecondary">{scoreWord(avg.score)}</T>
              </View>
            </View>
            <ShareRows ratio={avg.ratio} />
          </Group>
        ) : null}

        {todays.length ? (
          <>
            <SectionLabel>มื้อวันนี้</SectionLabel>
            <Group>
              {todays.map((m, i) => (
                <View key={m.id}>
                  {i > 0 ? <Separator /> : null}
                  <MealRow meal={m} />
                </View>
              ))}
            </Group>
          </>
        ) : (
          <Group style={styles.empty}>
            <T kind="title">ยังไม่มีมื้อของวันนี้</T>
            <T kind="body" tone="textSecondary">
              ถ่ายจานจากด้านบนให้เห็นทั้งจาน แอปจะแยกพื้นที่เป็นผัก ข้าว-แป้ง โปรตีน และผลไม้ แล้วเทียบกับจานสุขภาพ 2:1:1
              ผักครึ่งจาน ข้าว-แป้งหนึ่งในสี่ โปรตีนหนึ่งในสี่
            </T>
          </Group>
        )}
      </ScrollView>
      <CaptureBar />
    </View>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: Space.lg, width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center' },
  summary: { padding: Space.lg, gap: Space.lg },
  summaryTop: { flexDirection: 'row', alignItems: 'center', gap: Space.xl },
  summaryText: { flex: 1, gap: 2 },
  empty: { padding: Space.xl, gap: Space.md },
});
