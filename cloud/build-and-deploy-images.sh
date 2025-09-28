#!/bin/bash
# Build and deploy container images to Artifact Registry

set -e

PROJECT_ID=$1

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 PROJECT_ID"
    echo "Example: $0 my-iot-project"
    exit 1
fi

REGION="us-central1"
REPOSITORY="iot-pipeline-dev-repo"

echo "Building and deploying container images..."
echo "Project: $PROJECT_ID"
echo "Repository: $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY"

# Configure Docker for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build and push dashboard image
echo " Building dashboard image..."
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/dashboard:latest dashboard/
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/dashboard:latest

# Build and push data generator image
echo " Building data generator image..."
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/generator:latest generator/
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/generator:latest

# Update Cloud Run services with new images
echo " Deploying to Cloud Run..."

# Deploy dashboard
gcloud run deploy iot-pipeline-dev-dashboard \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/dashboard:latest \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID

# Update data generator job
gcloud run jobs replace iot-pipeline-dev-data-generator \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/generator:latest \
  --region $REGION

echo " Container images built and deployed successfully!"
echo ""
echo "Dashboard URL:"
gcloud run services describe iot-pipeline-dev-dashboard --region=$REGION --format="value(status.url)"
echo ""
echo "To start data generation:"
echo "gcloud run jobs execute iot-pipeline-dev-data-generator --region=$REGION"





