# Service account for the IoT pipeline
resource "google_service_account" "iot_pipeline_sa" {
  account_id   = "iot-pipeline-sa"
  display_name = "IoT Pipeline Service Account"
  description  = "Service account for IoT sensor data pipeline"
}

# BigQuery permissions
resource "google_project_iam_member" "bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

resource "google_project_iam_member" "bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

# Pub/Sub permissions
resource "google_project_iam_member" "pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

resource "google_project_iam_member" "pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

# Cloud Run permissions
resource "google_project_iam_member" "run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

# Workflows permissions
resource "google_project_iam_member" "workflows_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.iot_pipeline_sa.email}"
}

# Service account key for local development
resource "google_service_account_key" "iot_pipeline_key" {
  service_account_id = google_service_account.iot_pipeline_sa.name
}

# Output the service account email
output "service_account_email" {
  value = google_service_account.iot_pipeline_sa.email
}

# Output the service account key (base64 encoded)
output "service_account_key" {
  value     = google_service_account_key.iot_pipeline_key.private_key
  sensitive = true
}





