# KinWhy AI

ถ่ายรูปอาหาร 1 มื้อ แล้วดูว่าได้อาหารหลัก 5 หมู่ครบไหม และจานใกล้สูตร 2:1:1 แค่ไหน

**ใช้งานจริงได้ที่ https://kinwhy-ai.vercel.app** (เว็บ PWA กด "เพิ่มไปยังหน้าจอโฮม" ได้เหมือนแอป)

| ส่วน | อยู่ที่ | สถานะ |
|---|---|---|
| **เว็บ (ใช้งานหลักตอนนี้)** | `public/` → Vercel | ออนไลน์แล้ว โมเดล v4 รันในเบราว์เซอร์ รูปไม่ออกจากเครื่อง ไม่ต้องมีเซิร์ฟเวอร์ |
| แอปมือถือ (iOS + Android) | [mobile/](mobile/README.md) | พร้อม build แต่พักไว้ก่อน (ยังไม่ส่งสโตร์) |
| เซิร์ฟเวอร์วิเคราะห์ภาพ | `backend/` + `Dockerfile` | พร้อม deploy บน Cloud Run แต่ยังไม่ได้ deploy เพราะบัญชี Google Cloud ยังไม่มี billing ใช้กับแอปมือถือและตัวอุปกรณ์ห้อยคอ |

## ทำงานอย่างไร (เว็บ)

```
 เบราว์เซอร์ของผู้ใช้ (ทั้งหมดอยู่ในเครื่อง)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 1. รูปจากกล้องหรือคลังภาพ → ย่อเหลือด้านยาว 1280 px                         │
 │ 2. SegFormer-B2 v4 (int8, 55 MB, โหลดครั้งเดียวแล้วเก็บใน Cache API)          │
 │    รันใน Web Worker ด้วย onnxruntime-web (WASM หลายเธรด)                    │
 │    → ระบายสีทุกพิกเซล → พื้นที่ต่อคลาส                                        │
 │ 3. public/lib/plate.js: แปลงเป็นหมู่อาหาร คะแนน 2:1:1 คำแนะนำ                │
 │    (เป็นโค้ดชุดเดียวกับ backend/nutrition.py มีเทสต์เทียบให้ผลตรงกัน)            │
 │ 4. เก็บรูปและผลใน IndexedDB ของเครื่องนี้                                    │
 └──────────────────────────────────────────────────────────────────────┘
```

**หมู่ 5 ไขมัน** มองไม่เห็นเป็นพื้นที่ในภาพ จึงไม่อยู่ในสัดส่วน ส่วนเสริมอ่านเมนู/ปริมาณด้วย Claude (`backend/vlm.py`) ยังปิดอยู่และไม่ได้อยู่ในเว็บ

## หลักโภชนาการที่ใช้

- **อาหารหลัก 5 หมู่** (กระทรวงสาธารณสุข): หมู่ 1 เนื้อสัตว์ นม ไข่ ถั่ว · หมู่ 2 ข้าว แป้ง น้ำตาล เผือก มัน · หมู่ 3 ผัก · หมู่ 4 ผลไม้ · หมู่ 5 ไขมัน
- **ธงโภชนาการ** (กรมอนามัย) ต่อวัน ที่ 1,600 / 2,000 / 2,400 kcal: ข้าว-แป้ง 8/10/12 ทัพพี, ผัก 4/5/6 ทัพพี, ผลไม้ 3/4/5 ส่วน, เนื้อสัตว์ 6/9/12 ช้อนกินข้าว
- **หวาน มัน เค็ม 6:6:1**: น้ำตาล ≤ 6 ช้อนชา, น้ำมัน ≤ 6 ช้อนชา, เกลือ ≤ 1 ช้อนชา ต่อวัน
- **จาน 2:1:1** (สสส./กรมอนามัย): ผัก 2 ส่วน : ข้าว-แป้ง 1 ส่วน : เนื้อสัตว์ 1 ส่วน ของพื้นที่จาน ผลไม้กินแยกหลังอาหาร จึงไม่นับในอัตราส่วน
- คะแนน 2:1:1 = `100 × (1 − ผลรวมส่วนต่างจากเป้า / 150)` (150 คือส่วนต่างที่มากที่สุดที่เป็นไปได้)

ค่าทั้งหมดอยู่ใน [backend/nutrition.py](backend/nutrition.py)

## โครงสร้าง

```
public/                    เว็บ (static, Vercel)
  index.html, styles.css   หน้าเว็บ (liquid glass, PWA)
  app.js                   หน้าวันนี้ / ผลของมื้อ / ประวัติ / เกี่ยวกับ
  lib/plate.js             สัดส่วนจาน คะแนน 2:1:1 คำแนะนำ (พอร์ตจาก backend/nutrition.py)
  lib/segment.js           ต่อกับ worker, ย่อรูป, ภาพซ้อนสี
  lib/store.js             เก็บมื้ออาหารใน IndexedDB
  segmenter.worker.js      รันโมเดลในเบราว์เซอร์ (onnxruntime-web)
  sw.js, manifest          ใช้แบบออฟไลน์และติดตั้งเป็นแอป
  models/model.json        บอกว่าใช้ไฟล์โมเดลไหน (ตอนนี้ v4)
  privacy.html             นโยบายความเป็นส่วนตัว
mobile/                    แอป Expo (iOS + Android)
backend/                   FastAPI สำหรับแอป/อุปกรณ์: inference.py, nutrition.py, taxonomy.py, vlm.py
training/                  เทรน, pseudo-label, เครื่องมือ label ด้วย SAM, วัดผล
docs/                      เอกสารทางเทคนิค (PDF)
tests/                     pytest (รวมเทสต์ที่เทียบ plate.js กับ Python)
```

## รันในเครื่อง

```bash
pip install -r requirements.txt
```

```bash
python -m uvicorn backend.app:app --port 8000
```

เปิด http://localhost:8000 แล้วรันเทสต์ด้วย:

```bash
python -m pytest -q tests
```

## Deploy เว็บบน Vercel

เว็บเป็น static ทั้งหมด (`vercel.json` ตั้ง `outputDirectory: public` และ `.vercelignore` ให้อัปโหลดเฉพาะ `public/`)

```bash
npx vercel deploy --prod
```

`vercel.json` ใส่ header COOP/COEP ให้หน้าเว็บเป็น cross-origin isolated เพื่อให้ onnxruntime-web ใช้หลายเธรดได้ ถ้าเทรนโมเดลใหม่ ให้ตั้งชื่อไฟล์ .onnx ใหม่ทุกครั้ง แล้วแก้ `public/models/model.json` เพราะไฟล์โมเดลถูก cache แบบถาวรตามชื่อไฟล์

ทดสอบในเครื่อง: รัน `python -m uvicorn backend.app:app --port 8000` แล้วเปิด http://localhost:8000

## Deploy เซิร์ฟเวอร์บน Google Cloud Run (สำหรับแอปและตัวอุปกรณ์)

ต้องมี Google Cloud project ที่เปิด billing ไว้ และติดตั้ง `gcloud` แล้ว ไม่ต้องมี Docker ในเครื่อง เพราะ Cloud Build จะ build ให้

```bash
gcloud auth login
```

```bash
gcloud config set project <PROJECT_ID>
```

```bash
APP_ACCESS_CODE=<สุ่มข้อความยาวๆ> deploy/cloudrun.sh v3
```

สคริปต์นี้ทำอะไร:
- เก็บ `APP_ACCESS_CODE` ไว้ใน Secret Manager
- deploy ไปที่ region asia-southeast1 (สิงคโปร์ ใกล้ไทยที่สุด) ด้วย 2 vCPU / 2 GiB
- ตั้ง `min-instances=0` จึงไม่เสียเงินตอนไม่มีคนใช้ แต่คำขอแรกหลังว่างนานจะรอราว 10 วินาที
- จำกัด 20 รูปต่อนาทีต่อ IP
- ไม่เก็บรูปที่ส่งเข้ามา

เมื่อ deploy เสร็จ สคริปต์จะพิมพ์ URL ออกมา ให้เอาไปตั้งในแอปตาม [mobile/README.md](mobile/README.md)

| ตัวแปรของเซิร์ฟเวอร์ | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `APP_ACCESS_CODE` | (ว่าง = ไม่ต้องใช้รหัส) | คีย์ที่แอปต้องส่งมาใน header `X-Access-Code` |
| `RATE_LIMIT_PER_MIN` | 20 | จำนวนรูปต่อนาทีต่อ IP (0 = ไม่จำกัด) |
| `FOOD_MODEL_ONNX` / `FOOD_MODEL_META` | fp32 ของ `public/models/model.json` | เลือกไฟล์โมเดล |
| `PRELOAD_MODEL` | 0 (ใน image เป็น 1) | โหลดโมเดลตั้งแต่ตอนเปิดเซิร์ฟเวอร์ |

## เทรนโมเดลใหม่ (ต้องมี GPU + datasets_cache/)

```bash
pip install -r requirements-train.txt
```

```bash
python training/train_segformer_v3.py build
```

```bash
python training/train_segformer_v3.py train
```

ตั้งชื่อรุ่นด้วย `RUN_TAG` (เช่น `RUN_TAG=v5`) จะได้ `public/models/segformer_v5_int8.onnx` + `segformer_v5.json` และ `models/segformer_v5_fp32.onnx`
วัดผลด้วย `python training/evaluate.py models/segformer_v5_fp32.onnx public/models/segformer_v5.json` (รวมชุดทดสอบอาหารไทย) ถ้าดีกว่าเดิม ให้คัดลอก `segformer_v5.json` ไปเป็น `public/models/model.json`
**ทุกครั้งที่เทรนใหม่ ต้องใช้ชื่อไฟล์ .onnx ใหม่** เพราะเบราว์เซอร์ cache ไฟล์ตามชื่อแบบถาวร
รายละเอียดทั้งหมดอยู่ใน [docs/KinWhy-AI-Technical-Report.pdf](docs/KinWhy-AI-Technical-Report.pdf)

### คลาสของโมเดล (v3 ขึ้นไป)

`background, rice, other_starch, animal_protein, egg, plant_protein, vegetable, fruit, mixed_dish, sweets, liquid`

- อาหารคลุก (`mixed_dish` เช่น ข้าวผัด ผัดไทย) แบ่งเข้าหมู่ตามสัดส่วนที่ Claude ประเมิน ถ้าไม่มี Claude ใช้ค่าทั่วไป (แป้ง 65% โปรตีน 25% ผัก 10%)
- `liquid` (น้ำซุป เครื่องดื่ม) ไม่นับในสัดส่วนพื้นที่จาน

### ข้อมูลที่ใช้เทรน

| ชุดข้อมูล | จำนวน | mask |
|---|---|---|
| FoodSeg103 train | 4,982 | มนุษย์ติดป้าย |
| THFOOD-50 (อาหารไทย 50 เมนู) | 2,393 → 5,808 (v3b) | pseudo-label จากโมเดลรุ่นก่อน |
| อาหารไทยที่ตรวจแล้ว (v4) | 345 (ใส่ซ้ำ ×4) | SAM 2.1 ลากขอบ แล้วตรวจและแก้ทีละส่วน ([training/sam_label.py](training/sam_label.py)) |
| Nutrition5k | 888 | pseudo-label จากโมเดลรุ่นก่อน |
| Tiny-ImageNet (ไม่ใช่อาหาร) | 800 | ทั้งภาพเป็นพื้นหลัง |

## ข้อจำกัดที่ควรรู้

- ระบบวัดสัดส่วนพื้นที่ที่มองเห็น ไม่ใช่น้ำหนักหรือแคลอรี อาหารที่ซ้อนกันหรือจมในน้ำแกงจะถูกนับตามที่เห็นเท่านั้น
- ชุดทดสอบอาหารไทยมี 149 รูป (`datasets_cache/thai_sam_test.parquet`, `python training/evaluate.py ...` จะวัดให้อัตโนมัติ) ขอบภาพวาดด้วย SAM และป้ายตรวจด้วยมือ แต่ยังไม่ได้ให้นักโภชนาการตรวจซ้ำ
  บนชุดนี้ v4 คลาดเคลื่อน 2:1:1 เฉลี่ย 10.7 จุด (fp32) / 12.3 จุด (int8 ในเว็บ) ขณะที่ v3 อยู่ที่ 14-16 จุด และบน FoodSeg103 อยู่ที่ 8 จุด อาหารไทยยังยากกว่าชัดเจน
  เมนูที่ยังพลาดมาก: โจ๊ก หอยลายผัดพริกเผา เกี๊ยว · คลาสที่ยังอ่อน: เต้าหู้/ถั่ว น้ำซุป อาหารคลุก
- ชุดทดสอบ "ไม่ใช่อาหาร" เป็นภาพ Tiny-ImageNet ขนาด 64×64 ซึ่งต่างจากรูปถ่ายมือถือจริง ควรเก็บรูปจริงมาทดสอบเพิ่ม
- โค้ดรุ่นก่อน (v1/v2: Mask2Former, depth 2.5D) อยู่ใน git history ก่อน commit นี้
