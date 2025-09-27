# Minimal Outputs - Key information for next steps

output "project_id" {
  description = "Google Cloud Project ID"
  value       = var.project_id
}

output "pubsub_topic" {
  description = "Pub/Sub topic for sensor data"
  value       = google_pubsub_topic.sensor_data.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset for analytics"
  value       = google_bigquery_dataset.pipeline.dataset_id
}

output "dashboard_url" {
  description = "Cloud Run dashboard URL"
  value       = google_cloud_run_v2_service.dashboard.uri
}

output "dashboard_service_name" {
  description = "Cloud Run dashboard service name"
  value       = google_cloud_run_v2_service.dashboard.name
}

output "function_name" {
  description = "Cloud Function name"
  value       = google_cloudfunctions2_function.ingest_data.name
}

output "service_account_email" {
  description = "Pipeline service account email"
  value       = google_service_account.pipeline_sa.email
}

output "function_source_bucket" {
  description = "Name of the GCS bucket for Cloud Function source code"
  value       = google_storage_bucket.function_source.name
}

output "scheduler_job_name" {
  description = "Cloud Scheduler job for automated data generation"
  value       = google_cloud_scheduler_job.data_generator.name
}

output "live_demo_info" {
  description = "Live demo URLs for sharing with employers"
  value = <<-EOT
    LIVE DEMO READY FOR EMPLOYERS:
    
    Dashboard (always live): ${google_cloud_run_v2_service.dashboard.uri}
    Data Generator: Runs automatically every hour via Cloud Scheduler
    BigQuery Console: https://console.cloud.google.com/bigquery?project=${var.project_id}
    
    The demo generates realistic IoT sensor data hourly and displays:
    - Real-time temperature, humidity, and soil moisture readings
    - Interactive charts and anomaly detection
    - Historical trends and aggregated data
    
    Estimated cost: ~$10-15/month for always-on demo
  EOT
}

output "next_steps" {
  description = "What to do after Terraform deployment"
  value = <<-EOT
    Infrastructure deployed successfully!
    
    For immediate testing:
    1. Trigger scheduler manually:
       gcloud scheduler jobs run ${google_cloud_scheduler_job.data_generator.name} --location=${var.region}
    
    2. Check BigQuery for data:
       https://console.cloud.google.com/bigquery?project=${var.project_id}
    
    3. View live dashboard:
       ${google_cloud_run_v2_service.dashboard.uri}
    
    For portfolio sharing:
    - GitHub: [Your repo URL]
    - Live Demo: ${google_cloud_run_v2_service.dashboard.uri}
  EOT
}