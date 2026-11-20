/** Floating capture controls: the one primary action of the app. */
import * as Haptics from 'expo-haptics';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import { useState } from 'react';
import { Alert, Platform, Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Glass } from '@/components/glass';
import { T } from '@/components/ui';
import { Radius, Space } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { startAnalysis } from '@/lib/analyze';

const OPTS: ImagePicker.ImagePickerOptions = { mediaTypes: 'images', quality: 1, exif: false };

export const CAPTURE_BAR_SPACE = 140; // scroll padding so the last row clears the bar and tab bar

// Development on web only: lets an automated browser feed a photo URL without a file dialog.
if (__DEV__ && Platform.OS === 'web' && typeof window !== 'undefined') {
  (window as unknown as { __food5Analyze: (uri: string) => string }).__food5Analyze = (uri) => {
    const id = startAnalysis({ uri, width: 4000, height: 3000 });
    router.push({ pathname: '/meal/[id]', params: { id } });
    return id;
  };
}

export function CaptureBar() {
  const t = useTheme();
  const insets = useSafeAreaInsets();
  const [busy, setBusy] = useState(false);

  async function pick(source: 'camera' | 'library') {
    if (busy) return;
    setBusy(true);
    try {
      if (source === 'camera' && Platform.OS !== 'web') {
        const perm = await ImagePicker.requestCameraPermissionsAsync();
        if (!perm.granted) {
          Alert.alert('ยังไม่ได้อนุญาตกล้อง', 'เปิดสิทธิ์กล้องในการตั้งค่าของเครื่อง หรือเลือกรูปจากคลังแทน');
          return;
        }
      }
      const res = source === 'camera' ? await ImagePicker.launchCameraAsync(OPTS) : await ImagePicker.launchImageLibraryAsync(OPTS);
      if (res.canceled || !res.assets?.[0]) return;
      if (Platform.OS !== 'web') void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      const a = res.assets[0];
      const id = startAnalysis({ uri: a.uri, width: a.width, height: a.height });
      router.push({ pathname: '/meal/[id]', params: { id } });
    } finally {
      setBusy(false);
    }
  }

  // Native tab bars sit at the bottom; on web the floating tab bar takes ~76pt.
  const bottom = Platform.select({ ios: insets.bottom + 64, android: Space.lg, default: 96 });

  return (
    <View pointerEvents="box-none" style={[styles.dock, { bottom }]}>
      <Glass radius={Radius.pill} style={styles.bar} interactive>
        <Pressable
          onPress={() => pick('camera')}
          disabled={busy}
          accessibilityRole="button"
          accessibilityLabel="ถ่ายรูปมื้ออาหาร"
          style={({ pressed }) => [styles.primary, { backgroundColor: t.tint, opacity: pressed || busy ? 0.85 : 1 }]}>
          <SymbolView name={{ ios: 'camera.fill', android: 'photo_camera', web: 'photo_camera' }} size={20} tintColor={t.onTint} />
          <T kind="headline" style={{ color: t.onTint }}>ถ่ายมื้อนี้</T>
        </Pressable>
        <Pressable
          onPress={() => pick('library')}
          disabled={busy}
          accessibilityRole="button"
          accessibilityLabel="เลือกรูปจากคลังภาพ"
          style={({ pressed }) => [styles.secondary, pressed && { opacity: 0.6 }]}>
          <SymbolView name={{ ios: 'photo.on.rectangle', android: 'photo_library', web: 'photo_library' }} size={22} tintColor={t.text} />
        </Pressable>
      </Glass>
    </View>
  );
}

const styles = StyleSheet.create({
  dock: { position: 'absolute', left: 0, right: 0, alignItems: 'center' },
  bar: { flexDirection: 'row', alignItems: 'center', padding: 6, gap: 4 },
  primary: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, paddingHorizontal: Space.xl, height: 52, borderRadius: Radius.pill },
  secondary: { width: 52, height: 52, alignItems: 'center', justifyContent: 'center', borderRadius: Radius.pill },
});
