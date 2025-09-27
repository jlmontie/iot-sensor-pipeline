import json
import os
import random
import time
import math
import signal
import sys
import argparse
from datetime import datetime, timezone, timedelta


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


def jitter(value, noise_level=0.05):
    """Add realistic noise to sensor readings"""
    return value + random.gauss(0, noise_level * abs(value))


def generate_temperature(sensor_id, t_hour):
    """Generate realistic temperature with daily cycles"""
    base_temp = 20 + 8 * math.sin(2 * math.pi * t_hour / 24)
    sensor_offset = hash(sensor_id) % 10 - 5
    return base_temp + sensor_offset


def generate_humidity(sensor_id, t_hour):
    """Generate humidity inversely correlated with temperature"""
    temp = generate_temperature(sensor_id, t_hour)
    base_humidity = 80 - (temp - 15) * 2
    return max(20, min(95, base_humidity))


def generate_moisture(sensor_id, t_hour):
    """Generate soil moisture with realistic watering cycles using exponential decay"""
    # Watering cycle parameters (customized per sensor)
    sensor_hash = hash(sensor_id) % 1000
    period = 168 + (sensor_hash % 48)  # 7±2 day watering cycles
    tau = 60 + (sensor_hash % 40)      # Decay rate variation
    amplitude = 0.8 + (sensor_hash % 100) / 500  # Peak moisture level
    
    # Phase within watering cycle [0, period)
    phase = t_hour % period
    
    # Exponential decay from peak moisture after watering
    base_moisture = amplitude * math.exp(-phase / tau)
    
    # Add small daily variation (evaporation patterns)
    daily_variation = 0.05 * math.sin(2 * math.pi * t_hour / 24)
    
    # Sensor-specific offset for calibration differences
    sensor_offset = (sensor_hash % 50) / 1000
    
    moisture = base_moisture + daily_variation + sensor_offset
    
    # Ensure realistic bounds (soil never completely dry or oversaturated)
    return max(0.15, min(0.9, moisture))


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


def main():
    producer = MessageProducer()

    # Cost-optimized settings
    if CLOUD_MODE:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 4)]  # Only 3 sensors for cloud
        sleep_interval = 60  # 1 minute between batches (much cheaper!)
        print("Cloud mode: Using cost-optimized settings (3 sensors, 1 min intervals)")
        print("Estimated cost: ~$5-10/month")
    else:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 11)]  # 10 sensors for local
        sleep_interval = 1  # 1 second for local demo
        print("Local mode: Using full simulation (10 sensors, 1 sec intervals)")

    mode_text = "Cloud (Pub/Sub)" if CLOUD_MODE else "Local (Kafka)"
    print(f"Streaming sensor data in {mode_text} mode...")
    print(f"Simulating {len(sensors)} sensors. Press Ctrl+C to stop.")

    if CLOUD_MODE:
        print(f"Sending data every {sleep_interval} seconds to minimize costs")

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
            t_hour = elapsed_seconds / 3600  # Convert to hours for pattern generation
            current_timestamp = start_time + timedelta(seconds=elapsed_seconds)

            for sensor_id in sensors:
                temp = generate_temperature(sensor_id, t_hour)
                humidity = generate_humidity(sensor_id, t_hour)
                moisture = generate_moisture(sensor_id, t_hour)

                payload = {
                    "sensor_id": sensor_id,
                    "timestamp": current_timestamp.isoformat(),
                    "temperature": round(jitter(temp, 0.1), 2),
                    "humidity": round(jitter(humidity, 0.05), 1),
                    "soil_moisture": round(jitter(moisture, 0.01), 2),
                    # Add legacy fields for local compatibility
                    **(
                        {
                            "temperature_c": round(jitter(temp, 0.1), 2),
                            "humidity_pct": round(jitter(humidity, 0.05), 1),
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
