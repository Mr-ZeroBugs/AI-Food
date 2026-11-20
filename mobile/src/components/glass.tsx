/**
 * Glass is used only where something floats over the meal photo (actions, the result panel), so the
 * food stays visible through it. iOS 26 gets the system Liquid Glass material; older iOS, Android and
 * web get the platform blur, which is the closest system material there.
 */
import { BlurView } from 'expo-blur';
import { GlassView, isGlassEffectAPIAvailable, isLiquidGlassAvailable } from 'expo-glass-effect';
import type { ReactNode } from 'react';
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { Radius } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

const LIQUID = Platform.OS === 'ios' && isLiquidGlassAvailable() && isGlassEffectAPIAvailable();

type Props = {
  children?: ReactNode;
  style?: StyleProp<ViewStyle>;
  radius?: number;
  interactive?: boolean;
  /** 'clear' keeps more of the photo visible; 'regular' is for text-heavy panels. */
  variant?: 'regular' | 'clear';
};

export function Glass({ children, style, radius = Radius.lg, interactive, variant = 'regular' }: Props) {
  const t = useTheme();
  const shape = { borderRadius: radius, overflow: 'hidden' as const };

  if (LIQUID) {
    return (
      <GlassView style={[shape, style]} glassEffectStyle={variant} isInteractive={interactive}>
        {children}
      </GlassView>
    );
  }
  return (
    <View style={[shape, styles.edge, { borderColor: t.separator }, style]}>
      <BlurView
        style={StyleSheet.absoluteFill}
        intensity={variant === 'clear' ? 40 : 70}
        tint={Platform.OS === 'ios' ? 'systemChromeMaterial' : t.scheme}
        blurMethod="dimezisBlurViewSdk31Plus"
      />
      {/* Android/web blur is lighter than iOS materials: a tonal wash keeps text at 4.5:1 over busy photos. */}
      <View
        pointerEvents="none"
        style={[
          StyleSheet.absoluteFill,
          { backgroundColor: t.scheme === 'dark' ? 'rgba(26,27,24,0.55)' : 'rgba(255,255,255,0.58)' },
        ]}
      />
      {children}
    </View>
  );
}

export const hasLiquidGlass = LIQUID;

const styles = StyleSheet.create({
  edge: { borderWidth: StyleSheet.hairlineWidth },
});
