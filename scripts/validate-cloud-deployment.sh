#!/bin/bash
# Cloud deployment validation script

set -e

echo "VALIDATING CLOUD DEPLOYMENT READINESS"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "terraform/main.tf" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Check required files
echo "Checking required files..."
required_files=(
    "terraform/main.tf"
    "terraform/variables.tf" 
    "terraform/outputs.tf"
    "terraform/terraform.tfvars"
    "cloud/functions/main.py"
    "cloud/functions/requirements.txt"
    "src/analytics/api.py"
    "src/dashboard/app.py"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "$file"
    else
        echo "$file (missing)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "Missing required files. Please ensure all files are present."
    exit 1
fi

echo ""
echo "Validating Terraform configuration..."
cd terraform

# Check Terraform formatting
if terraform fmt -check > /dev/null 2>&1; then
    echo "Terraform formatting is correct"
else
    echo "Terraform formatting issues (will be auto-fixed)"
    terraform fmt
fi

# Validate Terraform configuration
if terraform validate > /dev/null 2>&1; then
    echo "Terraform configuration is valid"
else
    echo "Terraform configuration has errors:"
    terraform validate
    exit 1
fi

cd ..

echo ""
echo "Checking authentication and project setup..."

# Check if gcloud is installed
if command -v gcloud &> /dev/null; then
    echo "gcloud CLI is installed"
else
    echo "gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null 2>&1; then
    active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1)
    echo "Authenticated as: $active_account"
else
    echo "Not authenticated with gcloud. Run: gcloud auth login"
    exit 1
fi

# Check project configuration
project_from_tfvars=$(grep 'project_id' terraform/terraform.tfvars | cut -d'"' -f2)
current_gcloud_project=$(gcloud config get-value project 2>/dev/null || echo "")

echo ""
echo "📋 Project Configuration:"
echo "   Terraform tfvars: $project_from_tfvars"
echo "   Current gcloud:    $current_gcloud_project"

if [ "$project_from_tfvars" != "$current_gcloud_project" ]; then
    echo "Project mismatch detected"
    echo "   To fix: gcloud config set project $project_from_tfvars"
fi

echo ""
echo "Deployment readiness check:"

# Check if required APIs might be enabled (basic check)
echo "   Checking API access..."
if gcloud services list --enabled --filter="name:bigquery.googleapis.com" --format="value(name)" > /dev/null 2>&1; then
    echo "   Can access Google Cloud APIs"
else
    echo "   May need to enable APIs (will be done during deployment)"
fi

echo ""
echo "VALIDATION COMPLETE"
echo "==================="
echo ""
echo "Ready for cloud deployment!"
echo ""
echo "Next steps:"
echo "1. Set project: export GCP_PROJECT_ID=$project_from_tfvars"
echo "2. Deploy:      ./scripts/run-cloud-demo.sh"
echo "3. Test:        Check the dashboard URL from Terraform outputs"
echo ""
echo "The deployment will:"
echo "   - Create BigQuery dataset and table"
echo "   - Deploy Cloud Function for data processing"
echo "   - Deploy Streamlit dashboard on Cloud Run"
echo "   - Frontload BigQuery with historical data"
echo ""
