import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';

import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { loadStore, useStore } from '@/lib/store';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const ready = useStore((s) => s.ready);
  const c = Colors[scheme];
  const base = scheme === 'dark' ? DarkTheme : DefaultTheme;

  useEffect(() => {
    void loadStore();
  }, []);
  useEffect(() => {
    if (ready) void SplashScreen.hideAsync();
  }, [ready]);

  if (!ready) return null; // splash stays up until the saved settings and meals are loaded

  return (
    <ThemeProvider
      value={{ ...base, colors: { ...base.colors, background: c.ground, card: c.surface, text: c.text, primary: c.tint, border: c.separator } }}>
      <StatusBar style="auto" />
      <Stack screenOptions={{ headerBackButtonDisplayMode: 'minimal', contentStyle: { backgroundColor: c.ground } }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="meal/[id]"
          options={{ title: '', headerTransparent: true, headerTintColor: '#FFFFFF', headerShadowVisible: false }}
        />
      </Stack>
    </ThemeProvider>
  );
}
