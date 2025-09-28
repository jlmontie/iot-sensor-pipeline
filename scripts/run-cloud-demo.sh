#!/bin/bash
# Cloud deployment script

set -e

echo "Starting cloud IoT pipeline..."

# Check required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID environment variable is required"
    echo "Set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Set cloud environment variables for compatibility
export BQ_DATASET=${BQ_DATASET:-iot_pipeline}
export PUBSUB_TOPIC=${PUBSUB_TOPIC:-iot-pipeline-dev-sensor-data}

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

# Deploy infrastructure (Phase 1: Everything except Cloud Function)
echo "Deploying infrastructure with Terraform (Phase 1)..."
cd terraform

# Initialize Terraform
terraform init

# Create temporary files that exclude the Cloud Function
echo "Planning Phase 1 deployment (excluding Cloud Function)..."
cp main.tf main.tf.backup
cp outputs.tf outputs.tf.backup

# Comment out the Cloud Function resource temporarily
sed -i.tmp '/^resource "google_cloudfunctions2_function" "ingest_data"/,/^}$/{
s/^/# /
}' main.tf

# Comment out the Cloud Function output temporarily
sed -i.tmp '/^output "function_name"/,/^}$/{
s/^/# /
}' outputs.tf

terraform plan

echo "Applying Phase 1 configuration..."
if ! terraform apply -auto-approve; then
    echo ""
    echo "Phase 1 deployment failed. Common solutions:"
    echo "1. Resource conflicts: terraform destroy && terraform apply"
    echo "2. Permission issues: Check IAM roles in GCP Console"  
    echo "3. API not enabled: Enable required APIs in GCP Console"
    # Restore files on failure
    if [ -f main.tf.backup ]; then
        mv main.tf.backup main.tf
        rm -f main.tf.tmp
    fi
    if [ -f outputs.tf.backup ]; then
        mv outputs.tf.backup outputs.tf
        rm -f outputs.tf.tmp
    fi
    exit 1
fi

# Upload function source to the bucket created by Terraform
echo "Uploading function source to GCS..."
if terraform output function_source_bucket > /dev/null 2>&1; then
    BUCKET_NAME=$(terraform output -raw function_source_bucket)
    echo "Using function source bucket: $BUCKET_NAME"
    gsutil cp /tmp/function-source.zip gs://$BUCKET_NAME/
    echo "Function source uploaded successfully"
    
    # Restore the original files and apply Phase 2
    echo "Applying Phase 2: Creating Cloud Function..."
    mv main.tf.backup main.tf
    mv outputs.tf.backup outputs.tf
    rm -f main.tf.tmp outputs.tf.tmp  # Clean up sed backup files
    terraform apply -auto-approve
    echo "Cloud Function deployed successfully"
else
    echo "Warning: Function source bucket output not found in Terraform"
    # Restore files even on failure
    if [ -f main.tf.backup ]; then
        mv main.tf.backup main.tf
        rm -f main.tf.tmp
    fi
    if [ -f outputs.tf.backup ]; then
        mv outputs.tf.backup outputs.tf
        rm -f outputs.tf.tmp
    fi
    exit 1
fi
cd ..

# Install dependencies
echo "Installing Python dependencies..."
pip install -r src/requirements.txt

echo "Cloud environment ready!"
echo ""
echo "Building and deploying API and dashboard containers..."
echo "This builds your FastAPI service and Streamlit dashboard and deploys them to Cloud Run."

# Build and push both images
REGION="us-central1"
REPO_NAME="iot-demo-dev-repo"  # Must match Terraform: ${local.name_prefix}-repo
API_IMAGE="$REGION-docker.pkg.dev/$GCP_PROJECT_ID/$REPO_NAME/api:latest"
DASHBOARD_IMAGE="$REGION-docker.pkg.dev/$GCP_PROJECT_ID/$REPO_NAME/dashboard:latest"

echo "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker $REGION-docker.pkg.dev

echo "Building FastAPI service Docker image..."
docker build --platform linux/amd64 -f Dockerfile.api -t $API_IMAGE .

echo "Building Streamlit dashboard Docker image..."
docker build --platform linux/amd64 -f Dockerfile -t $DASHBOARD_IMAGE .

echo "Pushing API image to Artifact Registry..."
docker push $API_IMAGE

echo "Pushing dashboard image to Artifact Registry..."
docker push $DASHBOARD_IMAGE

echo "Updating Cloud Run services with new images..."
gcloud run services update iot-demo-dev-api \
  --image $API_IMAGE \
  --region $REGION \
  --platform managed

gcloud run services update iot-demo-dev-dashboard \
  --image $DASHBOARD_IMAGE \
  --region $REGION \
  --platform managed

echo "Frontloading BigQuery with historical data..."
echo "This populates BigQuery with 6 weeks of historical data (3 sensors) for immediate visualization."

DASHBOARD_MODE=cloud GCP_PROJECT_ID=$GCP_PROJECT_ID python3 scripts/frontload-cloud-data.py
echo ""
echo "DEPLOYMENT COMPLETE!"
echo "===================="
echo ""
echo "Your analytics service is now live in the cloud!"
echo ""
echo "========================================="
echo "YOUR ANALYTICS SERVICE IS READY!"
echo "========================================="
echo ""
cd terraform
API_URL=$(terraform output -raw api_url)
DASHBOARD_URL=$(terraform output -raw dashboard_url)
cd ..

echo "FASTAPI SERVICE URL:"
echo "$API_URL"
echo ""
echo "DASHBOARD URL:"
echo "$DASHBOARD_URL"
echo ""
echo "Copy and paste the dashboard URL into your browser:"
echo "$DASHBOARD_URL"
echo ""
echo "========================================="
echo "Next steps:"
echo "1. View dashboard: Open the dashboard URL above in your browser"
echo "2. Test API: Visit $API_URL/docs for interactive API documentation"
echo "3. Share with employers: The dashboard URL is your live demo!"
echo ""
echo "The system is fully automated - no manual data generation needed!"
echo "Your ForecastWater ML algorithm is now running as a production service."
echo ""
