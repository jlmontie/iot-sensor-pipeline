#!/bin/bash
# Deploy live demo with always-on dashboard and automated data generation

set -e

echo "Deploying live IoT demo for employer showcase..."

# Check required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID environment variable is required"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Set environment variables
export BQ_DATASET=${BQ_DATASET:-iot_demo_dev_pipeline}
export PUBSUB_TOPIC=${PUBSUB_TOPIC:-iot-demo-dev-sensor-data}

echo "Deploying to Google Cloud..."
echo "Project: $GCP_PROJECT_ID"
echo "Dataset: $BQ_DATASET"
echo "Topic: $PUBSUB_TOPIC"

# Prepare Cloud Function source code
echo "Preparing Cloud Function source code..."
mkdir -p /tmp/function-source
cp cloud/functions/main.py /tmp/function-source/
cp cloud/functions/requirements.txt /tmp/function-source/

# Create function source zip
echo "Creating function source archive..."
cd /tmp/function-source
zip -r ../function-source.zip .
cd - > /dev/null

# Deploy infrastructure
echo "Deploying infrastructure with Terraform..."
cd terraform

# Initialize and apply Terraform
terraform init
terraform apply -auto-approve

# Get outputs
DASHBOARD_SERVICE_NAME=$(terraform output -raw dashboard_service_name 2>/dev/null || echo "iot-demo-dev-dashboard")
FUNCTION_SOURCE_BUCKET=$(terraform output -raw function_source_bucket)

cd ..

# Upload function source
echo "Uploading Cloud Function source..."
gsutil cp /tmp/function-source.zip gs://$FUNCTION_SOURCE_BUCKET/

# Build and deploy dashboard image
echo "Building and deploying dashboard image..."
IMAGE_NAME="gcr.io/$GCP_PROJECT_ID/iot-dashboard:latest"

# Build Docker image
docker build -t $IMAGE_NAME .

# Push to Google Container Registry
docker push $IMAGE_NAME

# Update Cloud Run service with new image
echo "Updating Cloud Run service..."
gcloud run services update $DASHBOARD_SERVICE_NAME \
    --image $IMAGE_NAME \
    --region us-central1 \
    --platform managed

# Trigger scheduler job to generate initial data
echo "Triggering initial data generation..."
SCHEDULER_JOB=$(terraform output -raw scheduler_job_name -state=terraform/terraform.tfstate)
gcloud scheduler jobs run $SCHEDULER_JOB --location=us-central1 || echo "Scheduler job will run automatically"

echo ""
echo "Live demo deployment complete!"
echo ""

# Display live demo information
cd terraform
terraform output live_demo_info
cd ..

echo ""
echo "Cleanup temporary files..."
rm -rf /tmp/function-source*

echo ""
echo "Demo is now live and will generate data every hour automatically!"
