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

output "api_url" {
  description = "Cloud Run API service URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "api_service_name" {
  description = "Cloud Run API service name"
  value       = google_cloud_run_v2_service.api.name
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

output "container_repository" {
  description = "Artifact Registry repository for container images"
  value       = google_artifact_registry_repository.container_repo.name
}


output "live_demo_info" {
  description = "Live demo URLs for sharing with employers"
  value       = <<-EOT
    LIVE DEMO READY FOR EMPLOYERS:
    
    Dashboard (always live): ${google_cloud_run_v2_service.dashboard.uri}
    BigQuery Console: https://console.cloud.google.com/bigquery?project=${var.project_id}
    
    The demo showcases your ForecastWater ML algorithm with:
    - 6 weeks of historical sensor data (3 sensors)
    - Real-time watering predictions and analytics
    - Interactive dashboard with sensor status summary
    - Professional REST API with OpenAPI documentation
    
    Estimated cost: ~$5-10/month for always-on demo
  EOT
}

output "next_steps" {
  description = "What to do after Terraform deployment"
  value       = <<-EOT
    Infrastructure deployed successfully!
    
    Your analytics service is ready:
    1. View live dashboard:
       ${google_cloud_run_v2_service.dashboard.uri}
    
    2. Check BigQuery data:
       https://console.cloud.google.com/bigquery?project=${var.project_id}
    
    3. API documentation:
       The dashboard integrates with your ForecastWater API automatically
    
    For portfolio sharing:
    - GitHub: [Your repo URL]
    - Live Demo: ${google_cloud_run_v2_service.dashboard.uri}
  EOT
}