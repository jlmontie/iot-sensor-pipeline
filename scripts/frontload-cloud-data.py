#!/usr/bin/env python3
"""
Cloud data frontloading script for BigQuery
Generates 1000 hourly historical readings (~6 weeks) for 3 sensors
"""

import os
import sys
import time
import random
import math
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery


def jitter(value, noise_level=0.05):
    """Add realistic noise to sensor readings"""
    return value + random.gauss(0, noise_level * abs(value))


def generate_temperature(sensor_id, t_hour):
    """Generate temperature with daily and seasonal cycles"""
    base_temp = 20 + 5 * math.sin(2 * math.pi * t_hour / (24 * 30))  # Monthly cycle
    daily_variation = 3 * math.sin(2 * math.pi * t_hour / 24)
    sensor_offset = (hash(sensor_id) % 100) / 100
    return base_temp + daily_variation + sensor_offset


def generate_humidity(sensor_id, t_hour):
    """Generate humidity with daily and seasonal cycles"""
    base_humidity = 60 + 10 * math.sin(2 * math.pi * t_hour / (24 * 30))
    daily_variation = 5 * math.sin(2 * math.pi * t_hour / 24 + math.pi / 2)
    sensor_offset = (hash(sensor_id) % 100) / 200
    return base_humidity + daily_variation + sensor_offset


def generate_moisture(sensor_id, t_hour):
    """Generate soil moisture with realistic watering cycles using exponential decay"""
    # Watering cycle parameters (customized per sensor)
    sensor_hash = hash(sensor_id) % 1000
    period = 168 + (sensor_hash % 48)  # 7±2 day watering cycles
    tau = 60 + (sensor_hash % 40)  # Decay rate variation
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


def frontload_historical_data(client, table_ref, sensors, num_points=1000):
    """Insert historical data points into BigQuery for immediate dashboard visualization
    
    Args:
        num_points: Number of hourly readings to generate (default 1000 = ~6 weeks)
    """
    weeks = num_points / 24 / 7  # Convert points to weeks
    print(f"Frontloading {num_points} hourly historical data points (~{weeks:.1f} weeks)...")
    
    # Generate hourly data going back in time (6+ weeks)
    current_time = datetime.now(timezone.utc)
    time_interval = timedelta(hours=1)  # Hourly readings
    
    # Prepare batch data for BigQuery
    rows_to_insert = []
    
    for i in range(num_points):
        # Calculate time going backwards (hourly intervals)
        point_time = current_time - (time_interval * (num_points - i))
        
        # Calculate hours from start of simulation for pattern generation
        hours_from_start = (point_time.timestamp() - (current_time - timedelta(hours=num_points)).timestamp()) / 3600
        
        for sensor_id in sensors:
            temp = generate_temperature(sensor_id, hours_from_start)
            humidity = generate_humidity(sensor_id, hours_from_start)
            moisture = generate_moisture(sensor_id, hours_from_start)
            
            # Add noise
            temp = jitter(temp, 0.1)
            humidity = jitter(humidity, 0.05)
            moisture = jitter(moisture, 0.03)
            
            # Add to batch
            rows_to_insert.append({
                "sensor_id": sensor_id,
                "timestamp": point_time.isoformat(),
                "temperature": round(temp, 2),
                "humidity": round(humidity, 1),
                "soil_moisture": round(moisture, 3),
            })
        
        # Progress update
        if i % 50 == 0:
            days_back = (num_points - i) / 24
            print(f"Prepared {len(rows_to_insert)} historical readings... ({days_back:.1f} days back)")
    
    # Insert all data in batches for performance
    print(f"Inserting {len(rows_to_insert)} rows into BigQuery...")
    batch_size = 1000
    total_inserted = 0
    
    for i in range(0, len(rows_to_insert), batch_size):
        batch = rows_to_insert[i:i + batch_size]
        errors = client.insert_rows_json(table_ref, batch)
        
        if errors:
            print(f"Errors in batch {i//batch_size + 1}: {errors}")
        else:
            total_inserted += len(batch)
            print(f"Inserted batch {i//batch_size + 1}: {total_inserted}/{len(rows_to_insert)} rows")
    
    print(f"✅ Frontloaded {total_inserted} historical data points over {weeks:.1f} weeks")
    return total_inserted


def main():
    # Get configuration from environment
    project_id = os.getenv('GCP_PROJECT_ID')
    dataset_id = os.getenv('BQ_DATASET', 'iot_demo_dev_pipeline')
    table_id = os.getenv('BQ_TABLE', 'sensor_readings')
    
    if not project_id:
        print("Error: GCP_PROJECT_ID environment variable required")
        sys.exit(1)
    
    # Cloud mode uses 3 sensors for cost optimization
    sensors = [f"SENSOR-{i:03d}" for i in range(1, 4)]  # 3 sensors for cloud
    
    print(f"🚀 Starting Cloud data population with frontloading...")
    print(f"Environment: Cloud mode ({len(sensors)} sensors)")
    print(f"Target: {project_id}.{dataset_id}.{table_id}")
    print("This will first insert 1000 hourly historical points (~6 weeks)")
    print("Press Ctrl+C to stop")
    
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Check if table exists
        try:
            table = client.get_table(table_ref)
            print(f"Found existing table: {table.full_table_id}")
        except Exception as e:
            print(f"Error accessing table: {e}")
            print("Make sure the BigQuery table exists and you have proper permissions")
            sys.exit(1)
        
        # Clear any existing historical data (optional)
        clear_existing = input("Clear existing data first? (y/N): ").lower().strip()
        if clear_existing == 'y':
            query = f"DELETE FROM `{project_id}.{dataset_id}.{table_id}` WHERE sensor_id LIKE 'SENSOR-%'"
            job = client.query(query)
            job.result()  # Wait for completion
            print("Cleared existing sensor data")
        
        # Frontload historical data
        historical_count = frontload_historical_data(client, table_ref, sensors, 1000)
        
        print(f"\n📊 Historical data ready! Dashboard should now show rich data.")
        print(f"🌐 View in BigQuery: https://console.cloud.google.com/bigquery?project={project_id}")
        print(f"📈 Dashboard URL: Check your Cloud Run service")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
