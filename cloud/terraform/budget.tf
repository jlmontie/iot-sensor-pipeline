# Budget alert to control costs
resource "google_billing_budget" "iot_pipeline_budget" {
  billing_account = var.billing_account
  display_name    = "IoT Pipeline Budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "20"  # $20 monthly budget
    }
  }

  threshold_rules {
    threshold_percent = 0.5  # Alert at 50%
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 0.8  # Alert at 80%
    spend_basis       = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.0  # Alert at 100%
    spend_basis       = "CURRENT_SPEND"
  }

  all_updates_rule {
    monitoring_notification_channels = [
      google_monitoring_notification_channel.email.id
    ]
  }
}

# Email notification channel for budget alerts
resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification"
  type         = "email"

  labels = {
    email_address = var.notification_email
  }
}

# Variables
variable "billing_account" {
  description = "Billing account ID"
  type        = string
}

variable "notification_email" {
  description = "Email for budget notifications"
  type        = string
}





