import json
import os
import time
import signal
import sys
import argparse
from datetime import datetime, timezone, timedelta

# Import shared sensor simulation functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import generate_sensor_reading


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IoT Sensor Data Generator")
    parser.add_argument(
        "mode",
        choices=["local", "cloud"],
        help="Run in local mode (Kafka) or cloud mode (Pub/Sub)",
    )
    parser.add_argument("--project-id", help="GCP Project ID (required for cloud mode)")
    return parser.parse_args()


# Parse command line arguments
args = parse_args()
CLOUD_MODE = args.mode == "cloud"

# Configuration based on mode
if CLOUD_MODE:
    PROJECT_ID = args.project_id or os.getenv("GCP_PROJECT_ID")
    if not PROJECT_ID:
        print(
            "Error: --project-id required for cloud mode (or set GCP_PROJECT_ID env var)"
        )
        sys.exit(1)
    PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC", "iot-demo-dev-sensor-data")
    BQ_DATASET = os.getenv("BQ_DATASET", "iot_demo_dev_pipeline")
else:
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "sensor.readings")


# Sensor simulation functions moved to src/common/sensor_simulation.py


class MessageProducer:
    """Unified message producer for both Kafka and Pub/Sub"""

    def __init__(self):
        if CLOUD_MODE:
            from google.cloud import pubsub_v1

            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
            print(f"Pub/Sub publisher ready: {self.topic_path}")
        else:
            from confluent_kafka import Producer

            self.producer = Producer({"bootstrap.servers": KAFKA_BROKER})
            print(f"Kafka producer ready: {KAFKA_BROKER} -> {KAFKA_TOPIC}")

    def send_message(self, payload):
        """Send message to either Kafka or Pub/Sub"""
        message_data = json.dumps(payload).encode("utf-8")

        if CLOUD_MODE:
            future = self.publisher.publish(self.topic_path, message_data)
            return future.result()  # Wait for publish to complete
        else:
            self.producer.produce(
                KAFKA_TOPIC, value=message_data, callback=self.delivery_report
            )
            self.producer.poll(0)  # Trigger delivery callbacks

    def delivery_report(self, err, msg):
        """Kafka delivery callback"""
        if err is not None:
            print(f"Delivery failed: {err}")
        # Uncomment for verbose logging:
        # else:
        #     print(f"Delivered message to {msg.topic()} [{msg.partition()}]")

    def flush(self):
        """Flush pending messages"""
        if not CLOUD_MODE:
            self.producer.flush()


def get_simulation_time_offset():
    """
    Calculate the simulation time offset to continue from where frontloading left off.
    
    For local mode: Queries PostgreSQL to find the latest historical data timestamp
    For cloud mode: Uses standard 1000-hour offset (BigQuery query would be more complex)
    """
    if CLOUD_MODE:
        # For cloud mode, use standard frontload parameters
        FRONTLOAD_HOURS = 1000
        print(f"Cloud mode: Continuing simulation from hour {FRONTLOAD_HOURS}")
        return FRONTLOAD_HOURS
    
    # For local mode, try to detect the actual end of historical data
    
    # First, try to read simulation metadata file
    try:
        import json
        with open('.simulation_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        frontload_hours = metadata.get('frontload_hours', 1000)
        frontload_end_str = metadata.get('frontload_end_time')
        
        if frontload_end_str:
            from datetime import datetime, timezone
            frontload_end_time = datetime.fromisoformat(frontload_end_str.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            # Calculate time since frontloading completed
            time_diff = current_time - frontload_end_time
            hours_since_frontload = time_diff.total_seconds() / 3600
            
            # Continue from where frontloading left off
            simulation_offset = frontload_hours + hours_since_frontload
            
            print(f"Using simulation metadata: frontload ended {hours_since_frontload:.1f} hours ago")
            print(f"Continuing simulation from hour {simulation_offset:.1f} for seamless continuity")
            
            return simulation_offset
    except Exception as e:
        print(f"No simulation metadata found: {e}")
    
    # Fallback: Query database directly
    try:
        import psycopg2
        from datetime import datetime, timezone
        
        # Database connection (same as frontload script)
        DB_CONFIG = {
            'host': 'localhost',
            'port': 5433,
            'database': 'iot',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Find the latest timestamp in historical data
        cursor.execute("""
            SELECT MAX(event_time) FROM raw_sensor_readings 
            WHERE sensor_id LIKE 'SENSOR-%'
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0]:
            latest_historical_time = result[0]
            current_time = datetime.now(timezone.utc)
            
            # Calculate how many hours of historical data we have
            time_diff = current_time - latest_historical_time.replace(tzinfo=timezone.utc)
            hours_behind = time_diff.total_seconds() / 3600
            
            # The simulation should continue from where historical data ended
            # Historical data spans 1000 hours ending at latest_historical_time
            # So simulation time should be 1000 + hours_behind
            simulation_offset = 1000 + hours_behind
            
            print(f"Database query: historical data ending {hours_behind:.1f} hours ago")
            print(f"Continuing simulation from hour {simulation_offset:.1f} for seamless continuity")
            
            return simulation_offset
        else:
            print("No historical data found, starting simulation from hour 0")
            return 0
            
    except Exception as e:
        print(f"Could not detect historical data end time: {e}")
        print("Using default 1000-hour offset")
        return 1000

def main():
    producer = MessageProducer()

    # Cost-optimized settings
    if CLOUD_MODE:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 4)]  # Only 3 sensors for cloud
        sleep_interval = 60  # 1 minute between batches (much cheaper!)
        print("Cloud mode: Using cost-optimized settings (3 sensors, 1 min intervals)")
        print("Estimated cost: ~$5-10/month")
    else:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 6)]  # 5 sensors for local (matches frontload script)
        sleep_interval = 1  # 1 second for local demo
        print("Local mode: Using full simulation (5 sensors, 1 sec intervals)")

    mode_text = "Cloud (Pub/Sub)" if CLOUD_MODE else "Local (Kafka)"
    print(f"Streaming sensor data in {mode_text} mode...")
    print(f"Simulating {len(sensors)} sensors. Press Ctrl+C to stop.")

    if CLOUD_MODE:
        print(f"Sending data every {sleep_interval} seconds to minimize costs")

    # Get the time offset to continue from where frontloading left off
    simulation_hour_offset = get_simulation_time_offset()
    
    start_time = datetime.now(timezone.utc)
    message_count = 0

    def shutdown_handler(sig, frame):
        print(f"\nShutting down... Flushing {message_count} messages.")
        producer.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        while True:
            # Calculate elapsed time and current timestamp
            elapsed_seconds = time.time() - start_time.timestamp()
            # Add the simulation offset to continue from where frontloading left off
            t_hour = simulation_hour_offset + (elapsed_seconds / 3600)  # Continue simulation timeline
            current_timestamp = start_time + timedelta(seconds=elapsed_seconds)

            for sensor_id in sensors:
                # Generate sensor reading using shared simulation functions
                reading = generate_sensor_reading(sensor_id, t_hour, add_noise=True)

                payload = {
                    "sensor_id": sensor_id,
                    "timestamp": current_timestamp.isoformat(),
                    "temperature": reading['temperature'],
                    "humidity": reading['humidity'],
                    "soil_moisture": reading['soil_moisture'],
                    # Add legacy fields for local compatibility
                    **(
                        {
                            "temperature_c": reading['temperature'],
                            "humidity_pct": reading['humidity'],
                        }
                        if not CLOUD_MODE
                        else {}
                    ),
                }

                producer.send_message(payload)
                message_count += 1

                # Status update with cost awareness
                if CLOUD_MODE:
                    cost_per_message = 0.000040  # $40 per million messages
                    estimated_cost = message_count * cost_per_message
                    print(
                        f"Sent {message_count} messages (Est. cost: ${estimated_cost:.4f})"
                    )
                else:
                    if message_count % (len(sensors) * 10) == 0:
                        print(
                            f"Sent {message_count} messages ({message_count // len(sensors)} cycles)"
                        )

            time.sleep(sleep_interval)

    except Exception as e:
        print(f"Error: {e}")
        producer.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
