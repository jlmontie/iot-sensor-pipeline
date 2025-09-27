#!/usr/bin/env python3
"""
Temporary script to populate PostgreSQL with realistic sensor data
while we fix the Airflow DAG issue.
"""

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
    """Insert historical data points for immediate dashboard visualization"""
    print(f"Frontloading {num_points} historical data points...")
    
    # Generate data going back in time (last 24 hours)
    current_time = datetime.now(timezone.utc)
    time_interval = timedelta(minutes=1.44)  # ~1000 points over 24 hours
    
    count = 0
    for i in range(num_points):
        # Calculate time going backwards
        point_time = current_time - (time_interval * (num_points - i))
        t_hour = (point_time.timestamp() - current_time.timestamp()) / 3600 + 24  # Hours from 24h ago
        
        for sensor_id in sensors:
            temp = generate_temperature(sensor_id, t_hour)
            humidity = generate_humidity(sensor_id, t_hour)
            moisture = generate_moisture(sensor_id, t_hour)
            
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
        
        # Commit every 50 points for performance
        if i % 50 == 0:
            conn.commit()
            print(f"Inserted {count} historical readings...")
    
    conn.commit()
    print(f"✅ Frontloaded {count} historical data points")
    return count

def main():
    print("🚀 Starting PostgreSQL data population with frontloading...")
    print("This will first insert 1000 historical points, then continue with real-time data")
    print("Press Ctrl+C to stop")
    
    sensors = [f"SENSOR-{i:03d}" for i in range(1, 6)]  # 5 sensors
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Clear any existing test data first
        cursor.execute("DELETE FROM raw_sensor_readings WHERE sensor_id LIKE 'SENSOR-%'")
        conn.commit()
        print("Cleared existing sensor data")
        
        # Frontload historical data
        historical_count = frontload_historical_data(conn, cursor, sensors, 1000)
        
        print(f"\n📊 Historical data ready! Dashboard should now show rich data.")
        print(f"🔄 Starting real-time data generation...")
        
        # Continue with real-time data
        t0 = time.time()
        realtime_count = 0
        
        while True:
            t = time.time() - t0
            t_hour = t / 3600  # Convert to hours
            
            for sensor_id in sensors:
                temp = generate_temperature(sensor_id, t_hour)
                humidity = generate_humidity(sensor_id, t_hour)
                moisture = generate_moisture(sensor_id, t_hour)
                
                # Add noise
                temp = jitter(temp, 0.1)
                humidity = jitter(humidity, 0.05)
                moisture = jitter(moisture, 0.03)
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO raw_sensor_readings 
                    (sensor_id, event_time, temperature_c, humidity_pct, soil_moisture)
                    VALUES (%s, %s, %s, %s, %s)
                """, (sensor_id, datetime.now(timezone.utc), 
                      round(temp, 2), round(humidity, 1), round(moisture, 3)))
                
                realtime_count += 1
            
            conn.commit()
            total_count = historical_count + realtime_count
            print(f"📈 Total: {total_count} readings ({historical_count} historical + {realtime_count} realtime)")
            time.sleep(5)  # Insert every 5 seconds
            
    except KeyboardInterrupt:
        total_count = historical_count + realtime_count if 'historical_count' in locals() else 0
        print(f"\n🛑 Stopping... Total inserted: {total_count} readings")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
