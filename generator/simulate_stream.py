import json, os, random, time, math, uuid
from datetime import datetime, timezone
from kafka import KafkaProducer
import numpy as np
from scipy import signal

BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "sensor.readings")

# Moisture simulation parameters
WATER_SPIKE = 1.0
MOISTURE_MIN = 0.15
DRY_DURATION_HOURS = 5 * 24  # Time it takes to dry out
BASE_WATER_INTERVAL_HOURS = 5 * 24

# Per-sensor state
sensor_water_log = {}
sensor_water_interval = {}

def jitter(val, amt):
    return val + random.uniform(-amt, amt)

def generate_moisture(sensor_id, t_hour):
    # Initialize watering schedule if not already present
    if sensor_id not in sensor_water_log:
        # Random offset to avoid synchronized watering
        offset = random.uniform(-12, 12)
        sensor_water_log[sensor_id] = t_hour + offset
        sensor_water_interval[sensor_id] = BASE_WATER_INTERVAL_HOURS + random.uniform(-24, 24)

    last_watered = sensor_water_log[sensor_id]
    interval = sensor_water_interval[sensor_id]

    # If it's time to water
    if t_hour - last_watered >= interval:
        last_watered = t_hour
        sensor_water_log[sensor_id] = last_watered
        print(f"[{sensor_id}] Watered at {round(t_hour, 2)} hours")

    dt = t_hour - last_watered
    decay_factor = math.exp(-dt / DRY_DURATION_HOURS)
    moisture = MOISTURE_MIN + (WATER_SPIKE - MOISTURE_MIN) * decay_factor

    return moisture

def main():
    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    sensors = [f"SENSOR-{i:03d}" for i in range(1, 11)]
    print(f"Streaming to {BROKER} topic {TOPIC}... Ctrl+C to stop.")

    t0 = time.time()

    while True:
        t = time.time() - t0
        t_hour = t / 3600  # Convert to hours

        for sid in sensors:
            temp = 22 + 5 * math.sin(t / 180)
            humidity = 45 + 10 * math.sin(t / 240 + 1.0)
            moisture = generate_moisture(sid, t_hour)

            payload = {
                "timestamp": datetime.now(tz=timezone.utc).timestamp(),
                "sensor_id": sid,
                "temperature_c": round(jitter(temp, 0.5), 3),
                "humidity_pct": round(jitter(humidity, 2.0), 3),
                "soil_moisture": round(jitter(moisture, 0.03), 3)
            }

            producer.send(TOPIC, payload)

        producer.flush()
        time.sleep(1)

if __name__ == "__main__":
    main()
