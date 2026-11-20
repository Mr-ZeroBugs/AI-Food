/**
 * One meal: the photo fills the top of the screen with the segmentation laid over it, and a glass
 * panel rises over the photo's lower edge with the plate reading. The overlay can be toggled to check
 * what the model saw against the food itself.
 */
import { Image } from 'expo-image';
import { router, useLocalSearchParams } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Platform, Pressable, ScrollView, StyleSheet, useWindowDimensions, View } from 'react-native';
import Animated, { Easing, useAnimatedStyle, useSharedValue, withRepeat, withTiming, ReduceMotion } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Glass } from '@/components/glass';
import { PlateChart } from '@/components/plate-chart';
import { ShareRows } from '@/components/share-rows';
import { Group, T } from '@/components/ui';
import { Groups, MaxContentWidth, Radius, Space } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { runAnalysis } from '@/lib/analyze';
import { removeMeal, useStore } from '@/lib/store';
import { dayLabel, scoreWord, timeLabel } from '@/lib/stats';

export default function MealScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const meal = useStore((s) => s.meals.find((m) => m.id === id));
  const t = useTheme();
  const insets = useSafeAreaInsets();
  const { width, height } = useWindowDimensions();
  const [showMask, setShowMask] = useState(true);

  const r = meal?.result;
  const aspect = r?.image_size ? r.image_size[1] / r.image_size[0] : 0.75;
  const photoW = Math.min(width, MaxContentWidth);
  const photoH = Math.min(photoW * aspect, height * 0.62);

  // Scan line while the server works: the one motion moment of the app.
  const scan = useSharedValue(0);
  useEffect(() => {
    if (meal?.status === 'analyzing') {
      scan.value = 0;
      scan.value = withRepeat(withTiming(1, { duration: 1400, easing: Easing.inOut(Easing.cubic), reduceMotion: ReduceMotion.System }), -1, true);
    }
  }, [meal?.status, scan]);
  const scanStyle = useAnimatedStyle(() => ({ transform: [{ translateY: scan.value * (photoH - 2) }] }));

  if (!meal) {
    return (
      <View style={[styles.center, { backgroundColor: t.ground }]}>
        <T kind="body" tone="textSecondary">ไม่พบมื้อนี้ อาจถูกลบไปแล้ว</T>
      </View>
    );
  }

  function confirmDelete() {
    const go = () => {
      removeMeal(meal!.id);
      router.back();
    };
    if (Platform.OS === 'web') {
      if (window.confirm('ลบมื้อนี้ออกจากประวัติ?')) go();
      return;
    }
    Alert.alert('ลบมื้อนี้?', 'รูปและผลวิเคราะห์ของมื้อนี้จะถูกลบออกจากเครื่อง', [
      { text: 'ยกเลิก', style: 'cancel' },
      { text: 'ลบ', style: 'destructive', onPress: go },
    ]);
  }

  const food = r && r.is_food ? r : null;
  const plate = food?.plate.plate_211;

  return (
    <ScrollView style={{ flex: 1, backgroundColor: t.ground }} contentContainerStyle={{ paddingBottom: insets.bottom + Space.xxl }}>
      <View style={[styles.photoWrap, { height: photoH }]}>
        <Image source={{ uri: meal.photoUri }} style={StyleSheet.absoluteFill} contentFit="cover" transition={200} />
        {food?.overlay_png_b64 && showMask ? (
          <Image
            source={{ uri: `data:image/png;base64,${food.overlay_png_b64}` }}
            style={[StyleSheet.absoluteFill, { opacity: 0.45 }]}
            contentFit="fill"
            transition={350}
          />
        ) : null}
        {meal.status === 'analyzing' ? (
          <>
            <View style={[StyleSheet.absoluteFill, { backgroundColor: t.scrim }]} />
            <Animated.View style={[styles.scanLine, scanStyle]} />
          </>
        ) : null}

        {/* Photo controls float on the photo: glass, because the food stays visible through them. */}
        <View style={[styles.photoBar, { bottom: Space.lg + Space.xl }]}>
          {food?.overlay_png_b64 ? (
            <Glass radius={Radius.pill} variant="clear" interactive>
              <Pressable onPress={() => setShowMask((v) => !v)} accessibilityRole="switch" accessibilityState={{ checked: showMask }} style={styles.chip}>
                <SymbolView name={{ ios: showMask ? 'eye.fill' : 'eye.slash', android: showMask ? 'visibility' : 'visibility_off', web: showMask ? 'visibility' : 'visibility_off' }} size={18} tintColor={t.text} />
                <T kind="footnote" style={{ fontWeight: '600' }}>{showMask ? 'ซ่อนสีหมู่อาหาร' : 'แสดงสีหมู่อาหาร'}</T>
              </Pressable>
            </Glass>
          ) : <View />}
          <Glass radius={Radius.pill} variant="clear" interactive>
            <Pressable onPress={confirmDelete} accessibilityRole="button" accessibilityLabel="ลบมื้อนี้" style={styles.iconChip}>
              <SymbolView name={{ ios: 'trash', android: 'delete', web: 'delete' }} size={18} tintColor={t.text} />
            </Pressable>
          </Glass>
        </View>
      </View>

      <View style={styles.sheetWrap}>
        <Glass radius={Radius.xl} style={styles.sheet}>
          <T kind="footnote" tone="textSecondary">{dayLabel(meal.createdAt)} · {timeLabel(meal.createdAt)}</T>

          {meal.status === 'analyzing' ? (
            <View style={styles.state}>
              <ActivityIndicator color={t.tint} />
              <T kind="headline">กำลังแยกพื้นที่หมู่อาหาร</T>
              <T kind="callout" tone="textSecondary">ส่งรูปไปวิเคราะห์ที่เซิร์ฟเวอร์ ปกติใช้เวลาไม่กี่วินาที</T>
            </View>
          ) : null}

          {meal.status === 'error' ? (
            <View style={styles.state}>
              <T kind="headline" tone="danger">วิเคราะห์ไม่สำเร็จ</T>
              <T kind="callout" tone="textSecondary">{meal.error}</T>
              <Pressable onPress={() => runAnalysis(meal)} accessibilityRole="button" style={({ pressed }) => [styles.retry, { backgroundColor: t.tint, opacity: pressed ? 0.85 : 1 }]}>
                <T kind="headline" style={{ color: t.onTint }}>ลองอีกครั้ง</T>
              </Pressable>
            </View>
          ) : null}

          {r && !r.is_food ? (
            <View style={styles.state}>
              <T kind="title">ไม่พบจานอาหาร</T>
              <T kind="callout" tone="textSecondary">{r.message_th}</T>
            </View>
          ) : null}

          {food && plate && plate.score !== null ? (
            <>
              <View style={styles.head}>
                <PlateChart ratio={plate.ratio} size={92} />
                <View style={{ flex: 1, gap: 2 }}>
                  <T kind="headline" style={styles.verdict}>{scoreWord(plate.score)}</T>
                  <T kind="footnote" tone="textSecondary">ใกล้จาน 2:1:1 แค่ไหน (เต็ม 100)</T>
                </View>
                <T kind="figure" style={styles.score}>{Math.round(plate.score)}</T>
              </View>
              <ShareRows ratio={plate.ratio} />
            </>
          ) : null}

          {food && plate && plate.score === null ? (
            <View style={styles.state}>
              <T kind="headline">ไม่มีผัก ข้าว หรือโปรตีนให้เทียบ</T>
              <T kind="callout" tone="textSecondary">จานนี้มีแต่ผลไม้หรือเครื่องดื่ม จึงไม่ได้ให้คะแนน 2:1:1</T>
            </View>
          ) : null}
        </Glass>
      </View>

      {food ? (
        <View style={styles.below}>
          {food.advice.length ? (
            <Group style={styles.block}>
              <T kind="headline">คำแนะนำ</T>
              {food.advice.map((a) => (
                <View key={a} style={styles.bullet}>
                  <View style={[styles.bulletDot, { backgroundColor: t.tint }]} />
                  <T kind="body" style={{ flex: 1 }}>{a}</T>
                </View>
              ))}
            </Group>
          ) : null}

          {food.warnings.length ? (
            <View style={[styles.block, styles.warn, { backgroundColor: t.warningBg }]}>
              {food.warnings.map((w) => <T key={w} kind="callout" tone="warning">{w}</T>)}
            </View>
          ) : null}

          <Group style={styles.block}>
            <T kind="headline">สิ่งที่โมเดลเห็นในภาพ</T>
            {food.detected_classes.map((c) => (
              <View key={c.class} style={styles.detected}>
                <T kind="callout" style={{ flex: 1 }}>{c.name_th}</T>
                <T kind="callout" tone="textSecondary" style={{ fontVariant: ['tabular-nums'] }}>{c.pct_of_image}% ของภาพ</T>
              </View>
            ))}
            {food.plate.group_pct.g4_fruit > 0 ? (
              <View style={styles.detected}>
                <View style={[styles.bulletDot, { backgroundColor: Groups.fruit.color }]} />
                <T kind="footnote" tone="textSecondary" style={{ flex: 1 }}>ผลไม้ไม่นับใน 2:1:1 แต่นับเป็นหมู่ 4 ของวัน</T>
              </View>
            ) : null}
            <T kind="footnote" tone="textTertiary">
              สัดส่วนคิดจากพื้นที่ที่มองเห็นในภาพ ไม่ใช่น้ำหนัก อาหารที่ซ้อนกันหรือไขมันในน้ำมันปรุงจะมองไม่เห็น · โมเดล {food.seg.model ?? '-'}
            </T>
          </Group>
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: Space.xl },
  photoWrap: { width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center', overflow: 'hidden', backgroundColor: '#000' },
  scanLine: { position: 'absolute', left: 0, right: 0, top: 0, height: 2, backgroundColor: 'rgba(255,255,255,0.9)', shadowColor: '#fff', shadowOpacity: 0.9, shadowRadius: 8, shadowOffset: { width: 0, height: 0 } },
  photoBar: { position: 'absolute', left: Space.lg, right: Space.lg, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  chip: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, paddingHorizontal: Space.lg, minHeight: 44 },
  iconChip: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  sheetWrap: { marginTop: -Space.xl, paddingHorizontal: Space.md, width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center' },
  sheet: { padding: Space.xl, gap: Space.lg },
  head: { flexDirection: 'row', alignItems: 'center', gap: Space.lg },
  score: { fontSize: 44, lineHeight: 48 },
  verdict: { fontSize: 20, lineHeight: 26, fontWeight: '700' },
  state: { gap: Space.sm, alignItems: 'flex-start' },
  retry: { marginTop: Space.sm, paddingHorizontal: Space.xl, minHeight: 48, borderRadius: Radius.pill, justifyContent: 'center' },
  below: { paddingHorizontal: Space.lg, paddingTop: Space.lg, gap: Space.md, width: '100%', maxWidth: MaxContentWidth, alignSelf: 'center' },
  block: { padding: Space.lg, gap: Space.md },
  warn: { borderRadius: Radius.md },
  bullet: { flexDirection: 'row', gap: Space.md, alignItems: 'flex-start' },
  bulletDot: { width: 6, height: 6, borderRadius: 3, marginTop: 10 },
  detected: { flexDirection: 'row', alignItems: 'center', gap: Space.sm },
});
