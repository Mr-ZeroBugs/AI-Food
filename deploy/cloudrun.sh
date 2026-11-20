#!/usr/bin/env bash
# Deploy the KinWhy AI inference server to Google Cloud Run (builds remotely with Cloud Build; no local
# Docker needed).
#
#   gcloud auth login && gcloud config set project <PROJECT_ID>     # once
#   APP_ACCESS_CODE=<random string> deploy/cloudrun.sh [model=v3] [region=asia-southeast1]
#
# The access code is stored as a Secret Manager secret, never in the image or the command history of
# the service. Put the same value in the app build (EXPO_PUBLIC_API_KEY, see mobile/eas.json).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-v3}"
REGION="${2:-asia-southeast1}"   # Singapore: the closest Cloud Run region to Thailand
SERVICE="${SERVICE:-kinwhy-api}"
: "${APP_ACCESS_CODE:?set APP_ACCESS_CODE to the shared key the app will send}"

test -f "models/segformer_${MODEL}_fp32.onnx" || { echo "missing models/segformer_${MODEL}_fp32.onnx"; exit 1; }
test -f "public/models/segformer_${MODEL}.json" || { echo "missing public/models/segformer_${MODEL}.json"; exit 1; }

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

if gcloud secrets describe kinwhy-app-key >/dev/null 2>&1; then
  printf %s "$APP_ACCESS_CODE" | gcloud secrets versions add kinwhy-app-key --data-file=-
else
  printf %s "$APP_ACCESS_CODE" | gcloud secrets create kinwhy-app-key --data-file=- --replication-policy=automatic
fi
PROJECT_NUMBER="$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding kinwhy-app-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor >/dev/null

# Stage the chosen model at the fixed path the Dockerfile copies.
mkdir -p deploy/model
cp "models/segformer_${MODEL}_fp32.onnx" deploy/model/model_fp32.onnx
cp "public/models/segformer_${MODEL}.json" deploy/model/model.json

# 2 vCPU / 2 GiB: one fp32 SegFormer-B2 inference takes ~0.5 s on 2 vCPU.
# min-instances 0 = no cost when idle (the first request after idle waits ~10 s for a cold start);
# use --min-instances=1 around launch.
gcloud run deploy "$SERVICE"   --source .   --region "$REGION"   --cpu 2 --memory 2Gi --concurrency 4 --timeout 60   --min-instances 0 --max-instances 5   --allow-unauthenticated   --set-env-vars "RATE_LIMIT_PER_MIN=20,ALLOWED_ORIGINS=*"   --set-secrets "APP_ACCESS_CODE=kinwhy-app-key:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "Deployed: $URL"
curl -fsS "$URL/api/health" && echo
echo "Next: put this URL in mobile/eas.json (EXPO_PUBLIC_API_URL) for the production build."
