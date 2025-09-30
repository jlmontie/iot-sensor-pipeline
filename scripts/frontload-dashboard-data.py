#!/usr/bin/env python3
"""
Script to frontload PostgreSQL with 6 weeks of historical sensor data
for immediate dashboard visualization. Does NOT generate real-time data.
Use src/generator/simulate_stream.py for real-time data generation.
"""

import os
import sys
import time
import psycopg2
from datetime import datetime, timezone, timedelta

# Import shared sensor simulation functions
# Import sensor simulation from local scripts directory
from sensor_simulation import generate_sensor_reading

# Database connection using environment variables
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5433')),
    'database': os.getenv('POSTGRES_DB', 'iot'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Sensor simulation functions moved to src/common/sensor_simulation.py

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
            # Generate sensor reading using shared simulation functions
            reading = generate_sensor_reading(sensor_id, hours_from_start, add_noise=True)
            
            # Insert into database
            cursor.execute("""
                INSERT INTO raw_sensor_readings 
                (sensor_id, event_time, temperature_c, humidity_pct, soil_moisture)
                VALUES (%s, %s, %s, %s, %s)
            """, (sensor_id, point_time, 
                  reading['temperature'], reading['humidity'], reading['soil_moisture']))
            
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
        
        # Save simulation metadata for continuity
        current_time = datetime.now(timezone.utc)
        simulation_metadata = {
            'frontload_hours': 1000,
            'frontload_end_time': current_time.isoformat(),
            'sensors': sensors
        }
        
        try:
            import json
            with open('.simulation_metadata.json', 'w') as f:
                json.dump(simulation_metadata, f, indent=2)
            print("Saved simulation metadata for real-time continuity")
        except Exception as e:
            print(f"Could not save simulation metadata: {e}")
        
        print(f"\nHistorical data ready! Dashboard should now show rich data.")
        print(f"Frontloading complete. Use src/generator/simulate_stream.py for real-time data.")
        print(f"Run: python3 src/generator/simulate_stream.py local")
        print(f"Real-time generator will automatically continue from hour 1000 for seamless data")
            
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
