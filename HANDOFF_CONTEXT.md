# Handoff context (KinWhy AI)

Read README.md first; this file records where things stand and why, so decisions are not re-litigated.
Full technical detail (model, data, training, evaluation): docs/KinWhy-AI-Technical-Report.pdf.

## Product
KinWhy AI: photograph one meal, see its plate shares (vegetables / carbs / protein / fruit) against the
Thai 2:1:1 healthy plate. Owner handles software + AI; a friend builds a small neck-worn camera (hardware,
carried around, photo taken manually at meals). The owner wants both a competition submission and a product.

## Current state (2026-11-20)
- **Live:** https://kinwhy-ai.vercel.app (Vercel project `kinwhy-ai`, static, deployed from `public/` via
  `npx vercel deploy --prod`). Model v4 int8 runs in the browser; photos never leave the device; meals are in
  IndexedDB. Installable PWA, works offline after first load.
- **Model in use: v4** (SegFormer-B2, 11 classes + food gate). Thai test set (149 plates) 2:1:1 error:
  v4 fp32 10.66 / v4 int8 12.25 / v3 fp32 14.09. FoodSeg103 val: v4 8.07, v3 8.14. Training finished
  2026-11-19; nothing is training now.
- **Mobile app** (`mobile/`, Expo SDK 58, iOS + Android, liquid glass via expo-glass-effect): built and
  checked in the web preview, never run on a real phone. Store release postponed by the owner.
- **Inference server** (`backend/` + `Dockerfile` + `deploy/cloudrun.sh`, POST /api/analyze-image, fp32,
  rate limit, no photo storage): ready but **not deployed**. The owner's Google Cloud account has hit its
  project quota and its billing accounts are closed. Needed later for the mobile app and the wearable.
- Nothing has been committed to git in this rebuild (branch `v3-rebuild`).

## Decisions made with the owner
1. Output = plate-area shares + 2:1:1 score. Amounts in ธงโภชนาการ units need the VLM (backend/vlm.py),
   which is postponed/maybe dropped ("model first").
2. Web first, app stores later: the web needs no server and no payment card.
3. Name "KinWhy AI"; temporary icon = 2:1:1 plate (mobile/assets/images, public/icons).
4. API keys only via environment variables, never in chat or code.

## Thai data (the main lever for accuracy)
- `training/sam_label.py` + `training/thai_labels.json`: 494 THFOOD-50 photos labelled by SAM 2.1 regions
  with every region checked by hand (149 test / 345 train). Build with `python training/sam_label.py build`.
- Weakest: congee/joke, stir-fried shellfish, dumplings, tofu/beans (IoU ~1), soup (~3), mixed dishes.
- Best next step: photos from the real wearable camera, labelled the same way (test set + training).

## Environment notes
- Windows dev box, RTX 4060 8 GB, Python 3.13, torch 2.6 cu124, torchvision 0.21 (matched; 0.25 was broken).
- PowerShell blocks .ps1 scripts: call `gcloud.cmd`, not `gcloud`, in PowerShell.
- Node 20 works but Expo SDK 58 asks for >= 22.13.
- `models/` (checkpoints, fp32 ONNX) and `datasets_cache/` are gitignored and large; copy them separately.
