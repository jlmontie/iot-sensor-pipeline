# Minimal Variables - Only what's essential for the IoT pipeline demo

variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  validation {
    condition     = length(var.project_id) > 0
    error_message = "Project ID cannot be empty."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "iot-pipeline"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-central1"
}

# Budget and billing variables
variable "billing_account" {
  description = "Billing account ID for budget alerts"
  type        = string
  default     = ""
}

variable "notification_email" {
  description = "Email for budget notifications"
  type        = string
  default     = ""
}