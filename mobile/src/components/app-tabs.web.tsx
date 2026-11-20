/** Web preview only: a floating glass tab bar standing in for the native one. */
import { TabList, TabSlot, TabTrigger, Tabs, type TabTriggerSlotProps } from 'expo-router/ui';
import { SymbolView } from 'expo-symbols';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { Glass } from '@/components/glass';
import { Font, Radius, Space, Type } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

type Btn = TabTriggerSlotProps & { icon: 'restaurant' | 'calendar_month' | 'settings'; label: string };

function TabButton({ isFocused, icon, label, ref: _ref, ...props }: Btn) {
  const t = useTheme();
  const color = isFocused ? t.tint : t.textSecondary;
  return (
    <Pressable {...props} style={styles.btn} accessibilityRole="tab" accessibilityState={{ selected: isFocused }}>
      <SymbolView name={{ ios: 'circle', android: icon, web: icon }} size={24} tintColor={color} />
      <Text style={[Type.caption, { color, fontFamily: Font, fontWeight: isFocused ? '700' : '500' }]}>{label}</Text>
    </Pressable>
  );
}

export default function AppTabs() {
  return (
    <Tabs>
      <TabSlot style={{ height: '100%' }} />
      <TabList asChild>
        <View style={styles.dock} pointerEvents="box-none">
          <Glass radius={Radius.pill} style={styles.bar}>
            <TabTrigger name="index" href="/" asChild>
              <TabButton icon="restaurant" label="วันนี้" />
            </TabTrigger>
            <TabTrigger name="history" href="/history" asChild>
              <TabButton icon="calendar_month" label="ประวัติ" />
            </TabTrigger>
            <TabTrigger name="settings" href="/settings" asChild>
              <TabButton icon="settings" label="ตั้งค่า" />
            </TabTrigger>
          </Glass>
        </View>
      </TabList>
    </Tabs>
  );
}

const styles = StyleSheet.create({
  dock: { position: 'absolute', bottom: Space.lg, left: 0, right: 0, flexDirection: 'row', justifyContent: 'center' },
  bar: { flexDirection: 'row', paddingHorizontal: Space.sm, paddingVertical: 6, gap: Space.xs },
  btn: { minWidth: 76, minHeight: 48, alignItems: 'center', justifyContent: 'center', gap: 2, borderRadius: Radius.pill },
});
