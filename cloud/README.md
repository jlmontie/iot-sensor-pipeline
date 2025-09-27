# 🌟 IoT Sensor Pipeline - Google Cloud Edition

**Fully serverless, production-ready IoT data pipeline showcasing modern cloud-native data engineering.**

## 🏗️ Architecture Overview

```
Cloud Run Job → Pub/Sub → Cloud Functions → BigQuery → dbt Cloud → Cloud Run Dashboard
     ↑              ↑           ↑              ↑           ↑            ↑
Data Generator   Streaming   Real-time      Warehouse   Transform   Analytics
(Scheduled)      Messages    Ingestion      (Serverless) (Managed)   (Auto-scale)
```

## 💰 **Cost Breakdown** (~$10.50/month)

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Pub/Sub | 1M messages | $0.40 |
| Cloud Functions | 720 executions | $0.20 |
| BigQuery | 10GB queries | Free (1TB free tier) |
| Cloud Run Dashboard | Always on | ~$7.00 |
| Cloud Run Jobs | 1hr/day | ~$3.00 |
| **Total** | | **~$10.60** |

## 🚀 **Quick Deploy**

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker installed
- Billing account ID and notification email

### One-Command Deploy
```bash
./cloud/deploy.sh YOUR_PROJECT_ID YOUR_BILLING_ACCOUNT your@email.com
```

### Manual Setup
```bash
# 1. Set up project
gcloud projects create iot-sensor-pipeline-demo
gcloud config set project iot-sensor-pipeline-demo

# 2. Enable APIs
gcloud services enable pubsub.googleapis.com bigquery.googleapis.com run.googleapis.com

# 3. Deploy infrastructure
cd cloud/terraform
terraform init
terraform apply

# 4. Deploy services
docker build -t gcr.io/PROJECT_ID/dashboard cloud/dashboard/
gcloud run deploy iot-dashboard --image gcr.io/PROJECT_ID/dashboard
```

## 📊 **What's Included**

### **Streaming Data Processing**
- **Pub/Sub**: Serverless message queue (replaces Kafka)
- **Cloud Functions**: Real-time data ingestion (replaces Airflow consumer)
- **Cloud Workflows**: Pipeline orchestration (replaces Airflow DAG)

### **Data Warehouse & Analytics**
- **BigQuery**: Serverless analytics warehouse (replaces Postgres)
- **dbt Cloud**: Managed transformations with CI/CD
- **Partitioned Tables**: Cost-optimized storage and querying

### **Applications & Monitoring**
- **Cloud Run Dashboard**: Auto-scaling Streamlit app
- **Cloud Run Jobs**: Scheduled data generation
- **Budget Alerts**: Automated cost monitoring
- **IAM Security**: Least-privilege service accounts

## 🔧 **Development Workflow**

### **Local Development**
```bash
# Run dashboard locally (connects to cloud BigQuery)
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project-id
streamlit run cloud/dashboard/app.py
```

### **CI/CD Pipeline**
- **GitHub Actions**: Automated testing and deployment
- **Container Registry**: Secure image storage
- **Blue/Green Deployments**: Zero-downtime updates

## 🛡️ **Security & Compliance**

### **IAM & Security**
- Service accounts with minimal required permissions
- No hardcoded credentials
- VPC-native networking (optional)
- Cloud KMS encryption at rest

### **Cost Controls**
- Budget alerts at 50%, 80%, 100%
- Query cost limits in BigQuery
- Auto-scaling with min/max instances
- Data retention policies

## 📈 **Monitoring & Observability**

### **Built-in Monitoring**
- Cloud Run metrics (latency, errors, CPU)
- BigQuery slot usage and costs
- Pub/Sub message throughput
- Custom anomaly detection alerts

### **Dashboards**
- Real-time sensor readings with anomaly highlighting
- Cost tracking and budget utilization
- Pipeline execution metrics
- Data quality scorecards

## 🎯 **Interview Talking Points**

### **Cloud-Native Design**
- **Serverless First**: No infrastructure management, pay-per-use
- **Auto-Scaling**: Handles traffic spikes automatically
- **Cost Optimization**: Intelligent resource allocation

### **Production Readiness**
- **CI/CD Pipeline**: Automated testing, linting, deployment
- **Monitoring**: Comprehensive observability and alerting
- **Security**: IAM best practices, encrypted data
- **Disaster Recovery**: Multi-region backup strategies

### **Scalability**
- **Horizontal Scaling**: Cloud Run scales to zero and up to millions of requests
- **Data Partitioning**: BigQuery tables partitioned by date for performance
- **Streaming Architecture**: Real-time processing with Pub/Sub

## 🌐 **Live Demo URLs**

After deployment, you'll have:
- **Dashboard**: `https://iot-dashboard-xyz-uc.a.run.app`
- **BigQuery Console**: Data exploration and querying
- **Cloud Monitoring**: Real-time metrics and alerts
- **GitHub Actions**: CI/CD pipeline status

## 📝 **Next Steps**

1. **Add ML Pipeline**: Anomaly detection with Vertex AI
2. **Multi-Region**: Deploy across multiple zones
3. **Real-time Alerts**: SMS/Email notifications for anomalies
4. **Mobile App**: Flutter app consuming the API
5. **Edge Computing**: IoT Core integration

---

**💡 This architecture showcases enterprise-grade data engineering skills while keeping costs under $15/month - perfect for demonstrating cloud expertise to potential employers!**





