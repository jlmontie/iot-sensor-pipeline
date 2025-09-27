#!/usr/bin/env python3
"""
Script to frontload PostgreSQL with 6 weeks of historical sensor data
for immediate dashboard visualization. Does NOT generate real-time data.
Use src/generator/simulate_stream.py for real-time data generation.
"""

import os
import time
import random
import math
import psycopg2
from datetime import datetime, timezone, timedelta

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'iot',
    'user': 'postgres',
    'password': 'postgres'
}

def generate_temperature(sensor_id, t_hour):
    """Generate temperature with daily cycles"""
    base_temp = 20 + 5 * math.sin(2 * math.pi * t_hour / (24 * 30))
    daily_variation = 3 * math.sin(2 * math.pi * t_hour / 24)
    sensor_offset = (hash(sensor_id) % 100) / 100
    return base_temp + daily_variation + sensor_offset

def generate_humidity(sensor_id, t_hour):
    """Generate humidity with daily cycles"""
    base_humidity = 60 + 10 * math.sin(2 * math.pi * t_hour / (24 * 30))
    daily_variation = 5 * math.sin(2 * math.pi * t_hour / 24 + math.pi / 2)
    sensor_offset = (hash(sensor_id) % 100) / 200
    return base_humidity + daily_variation + sensor_offset

def generate_moisture(sensor_id, t_hour):
    """Generate soil moisture with realistic watering cycles using exponential decay"""
    sensor_hash = hash(sensor_id) % 1000
    period = 168 + (sensor_hash % 48)  # 7±2 day watering cycles
    tau = 60 + (sensor_hash % 40)      # Decay rate variation
    amplitude = 0.8 + (sensor_hash % 100) / 500  # Peak moisture level
    
    # Phase within watering cycle
    phase = t_hour % period
    
    # Exponential decay from peak moisture after watering
    base_moisture = amplitude * math.exp(-phase / tau)
    
    # Add small daily variation
    daily_variation = 0.05 * math.sin(2 * math.pi * t_hour / 24)
    
    # Sensor-specific offset
    sensor_offset = (sensor_hash % 50) / 1000
    
    moisture = base_moisture + daily_variation + sensor_offset
    return max(0.15, min(0.9, moisture))

def jitter(value, noise_level=0.05):
    """Add realistic noise to sensor readings"""
    return value + random.gauss(0, noise_level * abs(value))

def frontload_historical_data(conn, cursor, sensors, num_points=1000):
    """Insert historical data points for immediate dashboard visualization
    
    Args:
        num_points: Number of hourly readings to generate (default 1000 = ~6 weeks)
    """
    weeks = num_points / 24 / 7  # Convert points to weeks
    print(f"Frontloading {num_points} hourly historical data points (~{weeks:.1f} weeks)...")
    
    # Generate hourly data going back in time (6+ weeks)
    current_time = datetime.now(timezone.utc)
    time_interval = timedelta(hours=1)  # Hourly readings
    
    count = 0
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
            
            # Insert into database
            cursor.execute("""
                INSERT INTO raw_sensor_readings 
                (sensor_id, event_time, temperature_c, humidity_pct, soil_moisture)
                VALUES (%s, %s, %s, %s, %s)
            """, (sensor_id, point_time, 
                  round(temp, 2), round(humidity, 1), round(moisture, 3)))
            
            count += 1
        
        # Commit every 50 time points (50 hours worth of data) for performance
        if i % 50 == 0:
            conn.commit()
            days_back = (num_points - i) / 24
            print(f"Inserted {count} historical readings... ({days_back:.1f} days back)")
    
    conn.commit()
    print(f"Frontloaded {count} historical data points over {weeks:.1f} weeks")
    return count

def main():
    # Detect environment mode (cloud uses fewer sensors for cost optimization)
    cloud_mode = os.getenv('DASHBOARD_MODE', '').lower() == 'cloud' or os.getenv('CLOUD_MODE', '').lower() == 'true'
    
    if cloud_mode:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 4)]  # 3 sensors for cloud
        env_name = "Cloud"
    else:
        sensors = [f"SENSOR-{i:03d}" for i in range(1, 6)]  # 5 sensors for local
        env_name = "Local"
    
    print(f"Starting {env_name} historical data frontloading...")
    print(f"Environment: {env_name} mode ({len(sensors)} sensors)")
    print("This will insert 1000 hourly historical points (~6 weeks) for dashboard visualization")
    print("Press Ctrl+C to stop")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Clear any existing test data first
        cursor.execute("DELETE FROM raw_sensor_readings WHERE sensor_id LIKE 'SENSOR-%'")
        conn.commit()
        print("Cleared existing sensor data")
        
        # Frontload historical data
        historical_count = frontload_historical_data(conn, cursor, sensors, 1000)
        
        print(f"\nHistorical data ready! Dashboard should now show rich data.")
        print(f"✅ Frontloading complete. Use src/generator/simulate_stream.py for real-time data.")
        print(f"💡 Run: python3 src/generator/simulate_stream.py local")
            
    except KeyboardInterrupt:
        historical_count = historical_count if 'historical_count' in locals() else 0
        print(f"\nStopped. Inserted: {historical_count} historical readings")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
