# Budget alert to control costs (optional - only created if billing_account is provided)
resource "google_billing_budget" "iot_pipeline_budget" {
  count           = var.billing_account != "" ? 1 : 0
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
      google_monitoring_notification_channel.email[0].id
    ]
  }
}

# Email notification channel for budget alerts (optional)
resource "google_monitoring_notification_channel" "email" {
  count        = var.notification_email != "" ? 1 : 0
  display_name = "Email Notification"
  type         = "email"

  labels = {
    email_address = var.notification_email
  }
}

# Budget variables are now defined in variables.tf





