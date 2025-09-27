#!/usr/bin/env python3
"""
Simple debug consumer to test Kafka -> PostgreSQL flow
Bypasses Airflow completely for troubleshooting
"""

import json
import psycopg2
from confluent_kafka import Consumer, KafkaError
from datetime import datetime

def main():
    print("🔧 DEBUG: Direct Kafka -> PostgreSQL consumer")
    print("This bypasses Airflow to test the basic data flow")
    
    # Database connection (same as Airflow)
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5433,
        'database': 'iot',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    # Kafka configuration (using external port from host)
    KAFKA_CONFIG = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'debug-consumer',
        'auto.offset.reset': 'latest'  # Only get new messages
    }
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL")
        
        # Connect to Kafka
        consumer = Consumer(KAFKA_CONFIG)
        consumer.subscribe(['sensor.readings'])
        print("✅ Connected to Kafka, waiting for messages...")
        print("Start your data generator now!")
        
        message_count = 0
        
        while message_count < 10:  # Process up to 10 messages for testing
            msg = consumer.poll(timeout=2.0)
            
            if msg is None:
                print("⏳ Waiting for messages...")
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"❌ Kafka error: {msg.error()}")
                    break
            
            try:
                # Parse message
                data = json.loads(msg.value().decode('utf-8'))
                print(f"📥 Received: {data['sensor_id']} at {data['timestamp']}")
                
                # Insert into database (convert ISO timestamp to PostgreSQL timestamp)
                cursor.execute("""
                    INSERT INTO raw_sensor_readings(event_time, sensor_id, temperature_c, humidity_pct, soil_moisture, payload)
                    VALUES (%s::timestamp, %s, %s, %s, %s, %s)
                """, (
                    data["timestamp"], 
                    data["sensor_id"], 
                    data.get("temperature_c", data.get("temperature")), 
                    data.get("humidity_pct", data.get("humidity")),
                    data.get("soil_moisture"), 
                    json.dumps(data)
                ))
                
                conn.commit()
                message_count += 1
                print(f"✅ Inserted record #{message_count}")
                
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                print(f"Message content: {msg.value().decode('utf-8')}")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        if 'consumer' in locals():
            consumer.close()
        if 'conn' in locals():
            conn.close()
        print(f"\n🏁 Debug consumer finished. Processed {message_count} messages.")

if __name__ == "__main__":
    main()
