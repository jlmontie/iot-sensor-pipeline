#!/usr/bin/env python3
"""
Direct Kafka to PostgreSQL bridge for testing data flow
Bypasses Airflow to ensure data reaches the dashboard
"""

import json
import sys
import os
import psycopg2
from datetime import datetime, timezone

# Add project root to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Database connection
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5433,
        'database': 'iot',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    # Kafka configuration
    KAFKA_BROKER = 'localhost:9092'
    KAFKA_TOPIC = 'sensor.readings'
    
    print("Starting direct Kafka → PostgreSQL bridge...")
    print("This bypasses Airflow for immediate data flow testing")
    
    try:
        from confluent_kafka import Consumer, KafkaError
        
        # Configure Kafka consumer
        consumer_config = {
            'bootstrap.servers': KAFKA_BROKER,
            'group.id': 'postgres-direct-consumer',
            'auto.offset.reset': 'latest'  # Only consume new messages
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([KAFKA_TOPIC])
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print(f"✓ Connected to Kafka: {KAFKA_BROKER}")
        print(f"✓ Connected to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        print("✓ Consuming messages from topic:", KAFKA_TOPIC)
        print("\nWaiting for Kafka messages... (Press Ctrl+C to stop)")
        
        message_count = 0
        
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    continue
            
            try:
                # Parse the JSON message
                sensor_data = json.loads(msg.value().decode('utf-8'))
                
                # Insert into PostgreSQL
                cursor.execute("""
                    INSERT INTO raw_sensor_readings 
                    (sensor_id, event_time, temperature_c, humidity_pct, soil_moisture)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    sensor_data.get('sensor_id'),
                    datetime.fromisoformat(sensor_data.get('timestamp').replace('Z', '+00:00')),
                    sensor_data.get('temperature_c', sensor_data.get('temperature')),
                    sensor_data.get('humidity_pct', sensor_data.get('humidity')),
                    sensor_data.get('soil_moisture')
                ))
                
                conn.commit()
                message_count += 1
                
                if message_count % 10 == 0:
                    print(f"✓ Processed {message_count} messages")
                
            except Exception as e:
                print(f"Error processing message: {e}")
                print(f"Message content: {msg.value().decode('utf-8')}")
                
    except ImportError:
        print("Error: confluent_kafka not installed")
        print("Run: pip install confluent-kafka")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'consumer' in locals():
            consumer.close()
        if 'conn' in locals():
            conn.close()
        print(f"\nStopped. Processed {message_count} total messages.")

if __name__ == "__main__":
    main()
