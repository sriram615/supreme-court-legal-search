#!/usr/bin/env bash
# Deploy to Cloud Run. Run from the repo root. Needs the gcloud CLI logged in
# (`gcloud init`) with a project that has billing enabled -- Cloud Run
# requires a billing account on file even though normal usage stays inside
# the Always Free monthly quota.

set -euo pipefail

PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
REGION="us-central1"
SERVICE_NAME="legal-search-engine"
MEMORY="3Gi"
CPU="2"
TIMEOUT="300"

if [ -z "$PROJECT_ID" ]; then
  echo "No gcloud project set. Run: gcloud init"
  exit 1
fi

echo "== Using project: $PROJECT_ID =="

echo "== Enable required APIs (one-time, harmless if already enabled) =="
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

echo "== Stage the Dockerfile at repo root (gcloud expects it alongside the build context) =="
cp deploy/gcp/Dockerfile ./Dockerfile
trap 'rm -f ./Dockerfile' EXIT

IMAGE_TAG="us-central1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/$SERVICE_NAME:latest"

echo "== Building fresh container image with Cloud Build =="
gcloud builds submit --tag "$IMAGE_TAG" .

echo "== Deploying image to Cloud Run =="
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_TAG" \
  --region "$REGION" \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --timeout "$TIMEOUT" \
  --min-instances 0 \
  --max-instances 3 \
  --allow-unauthenticated \
  --set-env-vars "LLM_PROVIDER=groq,LLM_MODEL=openai/gpt-oss-20b"

echo ""
echo "== Service deployed. URL: =="
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)'

echo ""
echo "== Next: attach your Groq key as a secret, not a plain env var =="
echo "  echo -n 'your-new-groq-key' | gcloud secrets create groq-api-key --data-file=-"
echo "  gcloud run services update $SERVICE_NAME --region $REGION --set-secrets=GROQ_API_KEY=groq-api-key:latest"
