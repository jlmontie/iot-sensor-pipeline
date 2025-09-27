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

output "next_steps" {
  description = "What to do after Terraform deployment"
  value = <<-EOT
    Infrastructure deployed successfully!
    
    Next steps:
    1. Test the pipeline:
       gcloud pubsub topics publish ${google_pubsub_topic.sensor_data.name} --message='{"sensor_id":"test","temperature":23.5,"timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}'
    
    2. Check BigQuery for data:
       https://console.cloud.google.com/bigquery?project=${var.project_id}
    
    3. View dashboard:
       ${google_cloud_run_v2_service.dashboard.uri}
  EOT
}