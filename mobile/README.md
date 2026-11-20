# KinWhy AI: แอปมือถือ (Expo, iOS + Android)

ถ่ายรูปมื้ออาหาร แล้วส่งให้เซิร์ฟเวอร์ (`POST /api/analyze-image`) แยกพื้นที่เป็นผัก ข้าว-แป้ง โปรตีน และผลไม้
เทียบกับจาน 2:1:1 จากนั้นเก็บประวัติรายวันและรายสัปดาห์ไว้ในเครื่อง

## รันบนมือถือ (ไม่ต้องมี Mac)

1. เปิดเซิร์ฟเวอร์ที่โฟลเดอร์หลักของโปรเจกต์ ต้องใช้ `--host 0.0.0.0` เพื่อให้มือถือต่อเข้ามาได้

   ```bash
   python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```

   ถ้า Windows ถามเรื่อง firewall ให้กดอนุญาตสำหรับเครือข่ายส่วนตัว (Private)

2. เปิดแอป

   ```bash
   cd mobile
   npx expo start
   ```

3. บน iPhone หรือ Android ให้ติดตั้งแอป **Expo Go** แล้วสแกน QR ที่ขึ้นในหน้าต่างคำสั่ง
   (iPhone สแกนด้วยแอปกล้อง ส่วน Android สแกนในแอป Expo Go) มือถือกับคอมต้องอยู่ Wi-Fi วงเดียวกัน

แอปจะตั้งที่อยู่เซิร์ฟเวอร์เป็นเครื่องเดียวกับที่รัน `expo start` พอร์ต 8000 ให้เอง ถ้าต่อไม่ได้
ให้แก้ที่แท็บ **ตั้งค่า** แล้วกด "ทดสอบการเชื่อมต่อ"

ดูในเบราว์เซอร์ก็ได้ด้วย `npx expo start --web` (ไว้ตรวจหน้าตา ส่วน Liquid Glass จะเห็นของจริงบน iOS 26 เท่านั้น)

## ปล่อยแอปจริง (build บน cloud ด้วย EAS ไม่ต้องมี Mac)

1. Deploy เซิร์ฟเวอร์ตามขั้นตอนใน README หลัก จะได้ URL หน้าตา `https://kinwhy-api-xxxx.a.run.app`
2. ตั้งค่า environment ของ EAS ทำครั้งเดียว โดยใช้ค่าเดียวกับ `APP_ACCESS_CODE` ของเซิร์ฟเวอร์

   ```bash
   npx eas-cli@latest login
   ```

   ```bash
   npx eas-cli@latest init
   ```

   ```bash
   npx eas-cli@latest env:create --environment production --name EXPO_PUBLIC_API_URL --value https://kinwhy-api-xxxx.a.run.app --visibility plaintext
   ```

   ```bash
   npx eas-cli@latest env:create --environment production --name EXPO_PUBLIC_API_KEY --value <APP_ACCESS_CODE> --visibility sensitive
   ```

   ทำซ้ำแบบเดียวกันสำหรับ `--environment preview` ด้วย

3. Build ตัวทดสอบ (Android ได้ไฟล์ APK ส่งให้เพื่อนติดตั้งได้ทันที iPhone ต้องลงทะเบียนเครื่องก่อน)

   ```bash
   npx eas-cli@latest build --profile preview --platform all
   ```

4. Build ตัวจริงแล้วส่งเข้าสโตร์ ต้องมีบัญชี Apple Developer ($99/ปี) และ Google Play Console ($25 จ่ายครั้งเดียว)

   ```bash
   npx eas-cli@latest build --profile production --platform all
   ```

   ```bash
   npx eas-cli@latest submit --platform all
   ```

แอปที่ build พร้อม `EXPO_PUBLIC_API_URL` จะซ่อนหน้าตั้งค่าเซิร์ฟเวอร์จากผู้ใช้ และใช้ URL ที่ฝังไว้ตอน build เสมอ

**ต้องทำก่อนส่งสโตร์:**
- นโยบายความเป็นส่วนตัวอยู่ที่ https://kinwhy-ai.vercel.app/privacy.html (`PRIVACY_URL` ใน `src/lib/config.ts`) ก่อนส่งสโตร์ให้ใส่อีเมลติดต่อจริงในหน้านั้น
- ตอบแบบฟอร์มความเป็นส่วนตัวของสโตร์ (App Privacy / Data safety) ให้ตรงกับนโยบาย: ไม่มีบัญชีผู้ใช้ ไม่มีการติดตามการใช้งาน รูปส่งไปประมวลผลแล้วไม่ถูกเก็บ
- ถ่ายภาพหน้าจอแอปสำหรับหน้าสโตร์ (iPhone 6.9" และ Android phone)
- ไอคอนตอนนี้เป็นตัวชั่วคราว (จาน 2:1:1, สร้างจาก Pillow) ถ้ามีโลโก้จริงให้แทนไฟล์ใน `assets/images/` โดยใช้ขนาดเดิม

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `src/app/(tabs)/index.tsx` | วันนี้: จานเฉลี่ยของวัน รายการมื้อ ปุ่มถ่ายรูป |
| `src/app/(tabs)/history.tsx` | ประวัติ: 7 วันล่าสุดและมื้อแยกตามวัน |
| `src/app/(tabs)/settings.tsx` | ระดับพลังงาน 1600/2000/2400, ที่อยู่เซิร์ฟเวอร์, รหัสเข้าใช้งาน, ลบประวัติ |
| `src/app/meal/[id].tsx` | ผลของมื้อ: รูปพร้อมสีหมู่อาหาร, แผนภาพจาน, คำแนะนำ |
| `src/lib/analyze.ts` | ส่งรูปขึ้นเซิร์ฟเวอร์และบันทึกผล |
| `src/lib/store.ts` | เก็บมื้ออาหารและการตั้งค่าในเครื่อง (AsyncStorage) |
| `src/components/glass.tsx` | Liquid Glass ของระบบบน iOS 26; ระบบอื่นใช้ blur ของระบบแทน |
| `src/components/plate-chart.tsx` | แผนภาพจาน: สัดส่วนจริงของมื้อ และขีดตำแหน่งที่ 2:1:1 ควรเป็น |

## หลักการออกแบบ

- Liquid Glass ใช้เฉพาะจุดที่ลอยอยู่เหนือรูปอาหาร (ปุ่มถ่าย, ปุ่มบนรูป, แผงผลลัพธ์) และแถบแท็บของระบบ
  ส่วนรายการทั่วไปใช้พื้นผิวทึบแบบ inset grouped ให้อ่านง่าย
- ใช้ตัวควบคุมของระบบ ได้แก่ แท็บบาร์ native (iOS 26 เป็น Liquid Glass, Android เป็น Material) ปุ่มย้อนกลับของระบบ และ SF Symbols / Material Symbols
- สีหมู่อาหารเป็นชุดเดียวกับเว็บและ overlay จากเซิร์ฟเวอร์
- ต้องใช้ Node.js 22.13 ขึ้นไป (Expo SDK 58)

## ข้อจำกัดตอนนี้

- รูปและประวัติเก็บในเครื่องเท่านั้น ยังไม่มีบัญชีผู้ใช้และยังไม่ sync ข้ามเครื่อง ถ้าลบแอปหรือเปลี่ยนเครื่อง ประวัติจะหาย
- รูปถูกย่อเหลือด้านยาว 1280 px ก่อนส่ง (ประมาณ 250 KB) ต้องต่ออินเทอร์เน็ตตอนวิเคราะห์ ถ้าส่งไม่สำเร็จ มื้อนั้นจะยังเก็บไว้และกดลองใหม่ได้
- ยังไม่ได้ต่อกับตัวอุปกรณ์ห้อยคอ ตัวอุปกรณ์ส่งรูปเข้า `/api/analyze-image` แบบเดียวกับแอปได้เลย
