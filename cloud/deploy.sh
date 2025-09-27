#!/bin/bash
set -e

# IoT Pipeline Cloud Deployment Script
# Usage: ./deploy.sh PROJECT_ID BILLING_ACCOUNT_ID EMAIL

PROJECT_ID=$1
BILLING_ACCOUNT_ID=$2
EMAIL=$3

if [ -z "$PROJECT_ID" ] || [ -z "$BILLING_ACCOUNT_ID" ] || [ -z "$EMAIL" ]; then
    echo "Usage: $0 PROJECT_ID BILLING_ACCOUNT_ID EMAIL"
    echo "Example: $0 my-iot-project-123 012345-6789AB-CDEFGH user@example.com"
    exit 1
fi

echo "🚀 Deploying IoT Pipeline to Google Cloud..."
echo "Project: $PROJECT_ID"
echo "Billing: $BILLING_ACCOUNT_ID"
echo "Email: $EMAIL"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📡 Enabling APIs..."
gcloud services enable pubsub.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable workflows.googleapis.com
gcloud services enable scheduler.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create Artifact Registry repository
echo "📦 Creating Artifact Registry..."
gcloud artifacts repositories create iot-pipeline \
    --repository-format=docker \
    --location=us-central1 \
    --description="IoT Pipeline container images" || true

# Create Pub/Sub topic and subscription
echo "📨 Creating Pub/Sub resources..."
gcloud pubsub topics create sensor-readings || true
gcloud pubsub subscriptions create sensor-readings-sub \
    --topic=sensor-readings \
    --ack-deadline=60 || true

# Create BigQuery dataset and tables
echo "🏢 Setting up BigQuery..."
bq mk --dataset --location=US $PROJECT_ID:iot_pipeline || true
bq query --use_legacy_sql=false < cloud/bigquery/schema.sql

# Deploy using Terraform (optional)
if command -v terraform &> /dev/null; then
    echo "🏗️ Deploying infrastructure with Terraform..."
    cd cloud/terraform
    terraform init
    terraform plan -var="project_id=$PROJECT_ID" -var="billing_account=$BILLING_ACCOUNT_ID" -var="notification_email=$EMAIL"
    terraform apply -auto-approve -var="project_id=$PROJECT_ID" -var="billing_account=$BILLING_ACCOUNT_ID" -var="notification_email=$EMAIL"
    cd ../..
fi

# Build and deploy Cloud Function
echo "⚡ Deploying Cloud Function..."
gcloud functions deploy ingest-sensor-data \
    --gen2 \
    --runtime=python311 \
    --region=us-central1 \
    --source=cloud/functions/ingest_sensor_data \
    --entry-point=pubsub_trigger \
    --trigger-topic=sensor-readings \
    --memory=512MB \
    --timeout=60s

# Build and deploy dashboard
echo "📊 Building and deploying dashboard..."
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/dashboard:latest cloud/dashboard/
docker push us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/dashboard:latest

gcloud run deploy iot-dashboard \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/dashboard:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10

# Build and deploy data generator
echo "🤖 Building and deploying data generator..."
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/generator:latest cloud/generator/
docker push us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/generator:latest

gcloud run jobs create iot-data-generator \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/iot-pipeline/generator:latest \
    --region us-central1 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,PUBSUB_TOPIC=sensor-readings \
    --memory 512Mi \
    --cpu 1 \
    --max-retries 3 \
    --parallelism 1 || true

# Deploy workflow
echo "🔄 Deploying workflow..."
gcloud workflows deploy iot-pipeline \
    --source=cloud/workflows/iot_pipeline.yaml \
    --location=us-central1

# Create scheduler job
echo "⏰ Creating scheduler..."
gcloud scheduler jobs create http iot-pipeline-trigger \
    --schedule="0 */1 * * *" \
    --uri="https://workflowexecutions.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/workflows/iot-pipeline/executions" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{}' \
    --oauth-service-account-email="iot-pipeline-sa@$PROJECT_ID.iam.gserviceaccount.com" || true

echo "✅ Deployment complete!"
echo ""
echo "🌐 Dashboard URL:"
gcloud run services describe iot-dashboard --region=us-central1 --format="value(status.url)"
echo ""
echo "🚀 To start data generation:"
echo "gcloud run jobs execute iot-data-generator --region=us-central1"
echo ""
echo "💰 Monitor costs at: https://console.cloud.google.com/billing"





