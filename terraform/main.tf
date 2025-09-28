# Minimal IoT Pipeline - Terraform Configuration
# Showcases key cloud architecture concepts without production complexity

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.9"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Local values for consistent naming
locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# Enable required Google Cloud APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "pubsub.googleapis.com",
    "bigquery.googleapis.com",
    "cloudfunctions.googleapis.com",
    "run.googleapis.com",
    "eventarc.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com"
  ])

  service            = each.value
  disable_on_destroy = false
}

# Wait for APIs to be enabled
resource "time_sleep" "wait_for_apis" {
  depends_on      = [google_project_service.apis]
  create_duration = "30s"
}

# Random suffix for bucket name uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Artifact Registry repository for container images
resource "google_artifact_registry_repository" "container_repo" {
  location      = var.region
  repository_id = "${local.name_prefix}-repo"
  description   = "Container images for IoT pipeline"
  format        = "DOCKER"
  
  depends_on = [time_sleep.wait_for_apis]
}

# Service Account for the pipeline
resource "google_service_account" "pipeline_sa" {
  account_id   = "${local.name_prefix}-pipeline-sa"
  display_name = "IoT Pipeline Service Account"
  depends_on   = [time_sleep.wait_for_apis]
}

# IAM roles for the service account
resource "google_project_iam_member" "pipeline_roles" {
  for_each = toset([
    "roles/pubsub.subscriber",
    "roles/pubsub.publisher",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

# Pub/Sub topic for sensor data
resource "google_pubsub_topic" "sensor_data" {
  name       = "${local.name_prefix}-sensor-data"
  depends_on = [time_sleep.wait_for_apis]
}

# BigQuery dataset
resource "google_bigquery_dataset" "pipeline" {
  dataset_id = "iot_pipeline"
  location   = "US"

  depends_on = [time_sleep.wait_for_apis]
}

# BigQuery table for raw sensor data
resource "google_bigquery_table" "sensor_readings" {
  dataset_id = google_bigquery_dataset.pipeline.dataset_id
  table_id   = "raw_sensor_readings"

  schema = jsonencode([
    {
      name = "sensor_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "event_time"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "temperature_c"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "humidity_pct"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "soil_moisture"
      type = "FLOAT64"
      mode = "NULLABLE"
    }
  ])

  # Partition by date for performance
  time_partitioning {
    type  = "DAY"
    field = "event_time"
  }
  deletion_protection = false
}

# Storage bucket for function source code
resource "google_storage_bucket" "function_source" {
  name     = "${local.name_prefix}-function-source-${random_id.bucket_suffix.hex}"
  location = var.region

  depends_on    = [time_sleep.wait_for_apis]
  force_destroy = true
}

# Cloud Function for data ingestion
resource "google_cloudfunctions2_function" "ingest_data" {
  name     = "${local.name_prefix}-ingest-data"
  location = var.region

  build_config {
    runtime     = "python311"
    entry_point = "process_sensor_data"

    source {
      storage_source {
        bucket = google_storage_bucket.function_source.name
        object = "function-source.zip"
      }
    }
  }

  service_config {
    service_account_email = google_service_account.pipeline_sa.email
    environment_variables = {
      GCP_PROJECT_ID = var.project_id
      BQ_DATASET     = google_bigquery_dataset.pipeline.dataset_id
      BQ_TABLE       = google_bigquery_table.sensor_readings.table_id
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.sensor_data.id
  }
}

# Cloud Run service for FastAPI analytics service
resource "google_cloud_run_v2_service" "api" {
  name     = "${local.name_prefix}-api"
  location = var.region

  template {
    service_account = google_service_account.pipeline_sa.email

    containers {
      image = "gcr.io/cloudrun/hello" # Will be updated by deployment script

      ports {
        container_port = 8000
      }

      env {
        name  = "DASHBOARD_MODE"
        value = "cloud"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "BQ_DATASET"
        value = google_bigquery_dataset.pipeline.dataset_id
      }
      env {
        name  = "BQ_TABLE"
        value = google_bigquery_table.sensor_readings.table_id
      }
    }
  }

  depends_on = [time_sleep.wait_for_apis]
}

# Make API publicly accessible
resource "google_cloud_run_service_iam_member" "api_public" {
  location = google_cloud_run_v2_service.api.location
  service  = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Run service for dashboard
resource "google_cloud_run_v2_service" "dashboard" {
  name     = "${local.name_prefix}-dashboard"
  location = var.region

  template {
    service_account = google_service_account.pipeline_sa.email

    containers {
      image = "gcr.io/cloudrun/hello" # Will be updated by deployment script

      env {
        name  = "DASHBOARD_MODE"
        value = "cloud"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "BQ_DATASET"
        value = google_bigquery_dataset.pipeline.dataset_id
      }
      env {
        name  = "BQ_TABLE"
        value = google_bigquery_table.sensor_readings.table_id
      }
      env {
        name  = "API_BASE_URL"
        value = google_cloud_run_v2_service.api.uri
      }
    }
  }

  depends_on = [time_sleep.wait_for_apis]
}

# Make dashboard publicly accessible
resource "google_cloud_run_service_iam_member" "dashboard_public" {
  location = google_cloud_run_v2_service.dashboard.location
  service  = google_cloud_run_v2_service.dashboard.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Note: No automated data generation needed for analytics demo
# Historical data is loaded once via frontload-cloud-data.py script





