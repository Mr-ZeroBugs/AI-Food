import Constants from "expo-constants";
import * as WebBrowser from "expo-web-browser";
import { useState } from "react";
import {
  Alert,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import {
  Group,
  ScreenTitle,
  SectionLabel,
  Separator,
  T,
} from "@/components/ui";
import { Font, MaxContentWidth, Radius, Space, Type } from "@/constants/theme";
import { useTheme } from "@/hooks/use-theme";
import { IS_CONFIGURED_BUILD, PRIVACY_URL } from "@/lib/config";
import { clearMeals, updateSettings, useStore } from "@/lib/store";
import type { Settings } from "@/lib/types";

const LEVELS: { v: Settings["energyLevel"]; who: string }[] = [
  { v: 1600, who: "เด็ก 6-13 ปี ผู้หญิงวัยทำงาน ผู้สูงอายุ" },
  { v: 2000, who: "วัยรุ่น ผู้ชายวัยทำงาน" },
  { v: 2400, who: "ผู้ใช้แรงงานมาก นักกีฬา" },
];

type Health = { state: "idle" | "checking" | "ok" | "fail"; note?: string };

export default function SettingsScreen() {
  const t = useTheme();
  const insets = useSafeAreaInsets();
  const settings = useStore((s) => s.settings);
  const count = useStore((s) => s.meals.length);
  const [url, setUrl] = useState(settings.serverUrl);
  const [code, setCode] = useState(settings.accessCode);
  const [health, setHealth] = useState<Health>({ state: "idle" });

  async function check() {
    setHealth({ state: "checking" });
    try {
      const res = await fetch(`${url.replace(/\/+$/, "")}/api/health`);
      const body = (await res.json()) as {
        version?: string;
        access_code_required?: boolean;
      };
      setHealth({
        state: "ok",
        note: `เวอร์ชัน ${body.version ?? "?"}${body.access_code_required ? " · ต้องใช้รหัสเข้าใช้งาน" : ""}`,
      });
    } catch {
      setHealth({ state: "fail", note: "ต่อไม่ได้ ตรวจที่อยู่และว่าเซิร์ฟเวอร์เปิดอยู่" });
    }
  }

  function confirmClear() {
    if (Platform.OS === "web") {
      if (window.confirm(`ลบประวัติทั้งหมด ${count} มื้อ? กู้คืนไม่ได้`)) clearMeals();
      return;
    }
    Alert.alert(
      `ลบประวัติทั้งหมด ${count} มื้อ?`,
      "รูปและผลวิเคราะห์ในเครื่องจะหายทั้งหมด กู้คืนไม่ได้",
      [
        { text: "ยกเลิก", style: "cancel" },
        { text: "ลบทั้งหมด", style: "destructive", onPress: clearMeals },
      ],
    );
  }

  const input = [Type.body, styles.input, { color: t.text, fontFamily: Font }];

  return (
    <ScrollView
      style={{ backgroundColor: t.ground }}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={[
        styles.content,
        {
          paddingTop: insets.top + Space.lg,
          paddingBottom: insets.bottom + 120,
        },
      ]}
    >
      <ScreenTitle title="ตั้งค่า" />

      <SectionLabel>พลังงานต่อวัน (ธงโภชนาการ)</SectionLabel>
      <Group>
        {LEVELS.map((l, i) => {
          const on = settings.energyLevel === l.v;
          return (
            <View key={l.v}>
              {i > 0 ? <Separator /> : null}
              <Pressable
                onPress={() => updateSettings({ energyLevel: l.v })}
                accessibilityRole="radio"
                accessibilityState={{ selected: on }}
                style={({ pressed }) => [
                  styles.option,
                  pressed && { backgroundColor: t.separator },
                ]}
              >
                <View style={{ flex: 1 }}>
                  <T kind="headline" style={{ fontVariant: ["tabular-nums"] }}>
                    {l.v.toLocaleString("th-TH")} กิโลแคลอรี
                  </T>
                  <T kind="footnote" tone="textSecondary">
                    {l.who}
                  </T>
                </View>
                <View
                  style={[
                    styles.radio,
                    { borderColor: on ? t.tint : t.textTertiary },
                  ]}
                >
                  {on ? (
                    <View
                      style={[styles.radioDot, { backgroundColor: t.tint }]}
                    />
                  ) : null}
                </View>
              </Pressable>
            </View>
          );
        })}
      </Group>

      {IS_CONFIGURED_BUILD ? null : (
        <>
          <SectionLabel>เซิร์ฟเวอร์วิเคราะห์ภาพ (โหมดพัฒนา)</SectionLabel>
          <Group>
            <View style={styles.field}>
              <T kind="footnote" tone="textSecondary">
                ที่อยู่
              </T>
              <TextInput
                value={url}
                onChangeText={setUrl}
                onEndEditing={() => updateSettings({ serverUrl: url.trim() })}
                onBlur={() => updateSettings({ serverUrl: url.trim() })}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                placeholder="http://192.168.1.10:8000"
                placeholderTextColor={t.textTertiary}
                style={input}
              />
            </View>
            <Separator />
            <View style={styles.field}>
              <T kind="footnote" tone="textSecondary">
                รหัสเข้าใช้งาน (ถ้าเซิร์ฟเวอร์ตั้งไว้)
              </T>
              <TextInput
                value={code}
                onChangeText={setCode}
                onEndEditing={() => updateSettings({ accessCode: code })}
                onBlur={() => updateSettings({ accessCode: code })}
                secureTextEntry
                autoCapitalize="none"
                placeholder="ไม่มี"
                placeholderTextColor={t.textTertiary}
                style={input}
              />
            </View>
            <Separator />
            <Pressable
              onPress={check}
              accessibilityRole="button"
              style={({ pressed }) => [
                styles.option,
                pressed && { backgroundColor: t.separator },
              ]}
            >
              <T kind="headline" tone="tint" style={{ flex: 1 }}>
                {health.state === "checking" ? "กำลังทดสอบ…" : "ทดสอบการเชื่อมต่อ"}
              </T>
              {health.note ? (
                <T
                  kind="footnote"
                  tone={health.state === "ok" ? "textSecondary" : "danger"}
                >
                  {health.note}
                </T>
              ) : null}
            </Pressable>
          </Group>
        </>
      )}

      <SectionLabel>ข้อมูลในเครื่อง</SectionLabel>
      <Group>
        <Pressable
          onPress={confirmClear}
          disabled={!count}
          accessibilityRole="button"
          style={({ pressed }) => [
            styles.option,
            pressed && { backgroundColor: t.separator },
            !count && { opacity: 0.4 },
          ]}
        >
          <T kind="headline" tone="danger" style={{ flex: 1 }}>
            ลบประวัติทั้งหมด
          </T>
          <T kind="footnote" tone="textSecondary">
            {count} มื้อ
          </T>
        </Pressable>
      </Group>
      <SectionLabel>เกี่ยวกับ</SectionLabel>
      <Group>
        <Pressable
          onPress={() => WebBrowser.openBrowserAsync(PRIVACY_URL)}
          accessibilityRole="link"
          style={({ pressed }) => [
            styles.option,
            pressed && { backgroundColor: t.separator },
          ]}
        >
          <T kind="body" style={{ flex: 1 }}>
            นโยบายความเป็นส่วนตัว
          </T>
          <T kind="footnote" tone="textTertiary">
            เปิดในเบราว์เซอร์
          </T>
        </Pressable>
        <Separator />
        <View style={styles.option}>
          <T kind="body" style={{ flex: 1 }}>
            เวอร์ชัน
          </T>
          <T
            kind="body"
            tone="textSecondary"
            style={{ fontVariant: ["tabular-nums"] }}
          >
            {Constants.expoConfig?.version ?? "-"}
          </T>
        </View>
      </Group>

      <T kind="footnote" tone="textTertiary" style={styles.note}>
        รูปและประวัติเก็บไว้ในเครื่องนี้เท่านั้น
        รูปถูกส่งไปที่เซิร์ฟเวอร์ด้านบนเพื่อวิเคราะห์และไม่ถูกเก็บไว้ที่นั่น
        ผลเป็นการประมาณจากพื้นที่ในภาพ ใช้ประกอบการกินให้สมดุล ไม่ใช่คำแนะนำทางการแพทย์
      </T>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: Space.lg,
    width: "100%",
    maxWidth: MaxContentWidth,
    alignSelf: "center",
  },
  option: {
    flexDirection: "row",
    alignItems: "center",
    gap: Space.md,
    paddingHorizontal: Space.lg,
    paddingVertical: Space.md,
    minHeight: 56,
  },
  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    alignItems: "center",
    justifyContent: "center",
  },
  radioDot: { width: 12, height: 12, borderRadius: 6 },
  field: {
    paddingHorizontal: Space.lg,
    paddingTop: Space.md,
    paddingBottom: Space.sm,
    gap: 2,
  },
  input: { paddingVertical: Space.sm, minHeight: 44, borderRadius: Radius.sm },
  note: { marginTop: Space.lg, marginHorizontal: Space.lg },
});
