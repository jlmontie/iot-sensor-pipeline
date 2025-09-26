# 🏗️ Terraform Infrastructure

**Minimal, demo-friendly IoT pipeline infrastructure for Google Cloud**

## 🎯 **What's Deployed**

This Terraform configuration creates a streamlined IoT data pipeline with **13 resources**:

### **Core Pipeline**
- **Pub/Sub Topic**: Event-driven messaging for sensor data
- **Cloud Function**: Serverless data processing (Python 3.11)
- **BigQuery Dataset & Table**: Data warehouse with partitioned table
- **Cloud Run Service**: Containerized dashboard application

### **Supporting Infrastructure**
- **Service Account**: Secure authentication with minimal IAM roles
- **Storage Bucket**: Function source code storage
- **Google Cloud APIs**: Required services (Pub/Sub, BigQuery, etc.)

## 🚀 **Quick Deploy**

```bash
# 1. Configure your project
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id

# 2. Standard Terraform workflow
terraform init
terraform plan
terraform apply

# 3. Test the pipeline
gcloud pubsub topics publish iot-demo-dev-sensor-data \
  --message='{"sensor_id":"test","temperature":23.5,"timestamp":"2025-09-26T18:00:00Z"}'
```

## ⚙️ **Configuration**

### **Required Variables**
```hcl
# terraform.tfvars
project_id = "your-gcp-project-id"  # REQUIRED
```

### **Optional Variables**
```hcl
project_name = "iot-demo"     # Default: "iot-pipeline"
environment  = "dev"          # Default: "dev"
region      = "us-central1"   # Default: "us-central1"
```

## 📋 **Resources Created**

| Resource Type | Count | Purpose |
|---------------|-------|---------|
| Pub/Sub Topic | 1 | Message queue for sensor data |
| BigQuery Dataset | 1 | Data warehouse |
| BigQuery Table | 1 | Partitioned sensor readings table |
| Cloud Function | 1 | Data processing |
| Cloud Run Service | 1 | Dashboard application |
| Service Account | 1 | Authentication |
| IAM Bindings | 4 | Least-privilege permissions |
| Storage Bucket | 1 | Function source code |
| Google APIs | 7 | Required cloud services |

**Total: 13 resources** (vs 42 in the complex version)

## 🎨 **Design Principles**

- **Simplicity**: Easy to understand and explain
- **Standards**: Uses standard Terraform workflow
- **Security**: Service account with minimal required permissions
- **Performance**: BigQuery table partitioned by timestamp
- **Cost-Effective**: Serverless components with auto-scaling

## 🔍 **After Deployment**

Check the outputs for next steps:
```bash
terraform output
```

The pipeline will be ready to:
1. Accept sensor data via Pub/Sub
2. Process data with Cloud Function
3. Store data in BigQuery
4. Display results via Cloud Run dashboard

Perfect for demonstrating cloud architecture skills! 🚀

## Cleanup

To remove all resources and avoid ongoing charges:

```bash
terraform destroy
```

**Note**: This removes all cloud resources but preserves Terraform state files for potential future deployments.