/** System tab bar: Liquid Glass on iOS 26, Material navigation bar on Android. */
import { NativeTabs } from 'expo-router/native-tabs';

import { useTheme } from '@/hooks/use-theme';

export default function AppTabs() {
  const t = useTheme();
  return (
    <NativeTabs tintColor={t.tint} minimizeBehavior="onScrollDown">
      <NativeTabs.Trigger name="index">
        <NativeTabs.Trigger.Label>วันนี้</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf={{ default: 'fork.knife.circle', selected: 'fork.knife.circle.fill' }} md="restaurant" />
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="history">
        <NativeTabs.Trigger.Label>ประวัติ</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf="calendar" md="calendar_month" />
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="settings">
        <NativeTabs.Trigger.Label>ตั้งค่า</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf="gearshape" md="settings" />
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}
