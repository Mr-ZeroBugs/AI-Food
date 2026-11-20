/** Each plate group's share against its 2:1:1 target, as a bar with a target mark. */
import { StyleSheet, Text, View } from 'react-native';

import { Font, Groups, PLATE_KEYS, Space, Type } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import type { PlateRatio } from '@/lib/types';

export function ShareRows({ ratio }: { ratio: PlateRatio }) {
  const t = useTheme();
  return (
    <View style={styles.list}>
      {PLATE_KEYS.map((k) => {
        const g = Groups[k];
        const v = Math.max(0, Math.min(100, ratio[k]));
        const gap = Math.round(v - g.target);
        return (
          <View key={k} style={styles.row}>
            <View style={styles.labels}>
              <View style={[styles.dot, { backgroundColor: g.color }]} />
              <Text style={[Type.callout, styles.name, { color: t.text, fontFamily: Font }]}>{g.label}</Text>
              <Text style={[Type.callout, styles.value, { color: t.text, fontFamily: Font }]}>{Math.round(v)}%</Text>
              <Text style={[Type.footnote, styles.target, { color: t.textSecondary, fontFamily: Font }]}>
                {Math.abs(gap) < 3 ? `ตรงเป้า ${g.target}%` : `เป้า ${g.target}% · ${gap > 0 ? 'เกิน' : 'ขาด'} ${Math.abs(gap)}`}
              </Text>
            </View>
            <View style={[styles.track, { backgroundColor: t.separator }]}>
              <View style={[styles.fill, { width: `${v}%`, backgroundColor: g.color }]} />
              <View style={[styles.mark, { left: `${g.target}%`, backgroundColor: t.text }]} />
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  list: { gap: Space.md },
  row: { gap: 6 },
  labels: { flexDirection: 'row', alignItems: 'center', gap: Space.sm },
  dot: { width: 10, height: 10, borderRadius: 5 },
  name: { fontWeight: '600' },
  value: { fontVariant: ['tabular-nums'], fontWeight: '600' },
  target: { marginLeft: 'auto', fontVariant: ['tabular-nums'] },
  track: { height: 8, borderRadius: 4, overflow: 'visible' },
  fill: { height: 8, borderRadius: 4 },
  mark: { position: 'absolute', top: -3, width: 2, height: 14, borderRadius: 1, marginLeft: -1 },
});
