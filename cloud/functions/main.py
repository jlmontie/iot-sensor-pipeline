import functions_framework
import json
import base64
import os
from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def process_sensor_data(cloud_event):
    """
    Cloud Function triggered by Pub/Sub to process IoT sensor data.
    This function is designed for Cloud Functions (2nd gen) with Pub/Sub triggers.
    Updated: Fixed GCP_PROJECT_ID environment variable issue.
    """
    try:
        # The Pub/Sub message is in the 'data' attribute of the CloudEvent
        pubsub_message_data = base64.b64decode(
            cloud_event.data["message"]["data"]
        ).decode("utf-8")
        sensor_data = json.loads(pubsub_message_data)

        logger.info(f"Received sensor data: {sensor_data}")

        # Get environment variables
        dataset_id = os.environ.get("BQ_DATASET", "iot_pipeline")
        table_id = os.environ.get("BQ_TABLE", "raw_sensor_readings")
        project_id = os.environ.get("GCP_PROJECT_ID")

        logger.info(
            f"Environment variables: BQ_DATASET={dataset_id}, BQ_TABLE={table_id}, GCP_PROJECT_ID={project_id}"
        )

        if not project_id:
            # Try to get project from the cloud event source
            try:
                project_id = (
                    cloud_event.get("source", "").split("/")[1]
                    if "/" in cloud_event.get("source", "")
                    else None
                )
                logger.info(f"Fallback project ID from cloud event: {project_id}")
            except Exception as e:
                logger.error(f"Error extracting project from cloud event: {e}")

        if not project_id:
            logger.error(
                "Could not determine GCP project ID from environment or cloud event"
            )
            logger.error(f"Available env vars: {list(os.environ.keys())}")
            return

        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(dataset_id).table(table_id)

        # Prepare data for BigQuery insertion
        rows_to_insert = [
            {
                "sensor_id": sensor_data.get("sensor_id"),
                "event_time": sensor_data.get("timestamp"),
                "temperature_c": sensor_data.get(
                    "temperature_c", sensor_data.get("temperature")
                ),
                "humidity_pct": sensor_data.get(
                    "humidity_pct", sensor_data.get("humidity")
                ),
                "soil_moisture": sensor_data.get("soil_moisture"),
            }
        ]

        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            logger.error(f"Errors inserting rows into BigQuery: {errors}")
        else:
            logger.info(
                f"Successfully inserted sensor data for {sensor_data.get('sensor_id')}"
            )

    except Exception as e:
        logger.error(f"Error processing Pub/Sub message: {e}", exc_info=True)
