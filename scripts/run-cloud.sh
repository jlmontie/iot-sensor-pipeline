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

# Deploy infrastructure (Phase 1: Everything except Cloud Function)
echo "Deploying infrastructure with Terraform (Phase 1)..."
cd terraform

# Initialize Terraform
terraform init

# Create temporary files that exclude the Cloud Function
echo "📋 Planning Phase 1 deployment (excluding Cloud Function)..."
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
echo "Next steps:"
echo "1. Frontload historical data: DASHBOARD_MODE=cloud GCP_PROJECT_ID=$GCP_PROJECT_ID python3 scripts/frontload-cloud-data.py"
echo "2. Start data generator:      python3 src/generator/simulate_stream.py cloud --project-id $GCP_PROJECT_ID"
echo "3. Launch dashboard:          DASHBOARD_MODE=cloud GCP_PROJECT_ID=$GCP_PROJECT_ID streamlit run src/dashboard/enhanced_app.py"
echo "4. Test with Pub/Sub:         gcloud pubsub topics publish $PUBSUB_TOPIC --message='{\"sensor_id\":\"test\",\"temperature\":23.5}'"
echo ""
echo "💡 The frontloading step populates BigQuery with 6 weeks of historical data (3 sensors)"
echo "   so your dashboard shows rich visualizations immediately!"
echo ""
