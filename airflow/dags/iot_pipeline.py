from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
# Uncomment for BigQuery integration:
# from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
# from google.cloud import bigquery
import json, os, time
import psycopg2
from confluent_kafka import Consumer, KafkaError

POSTGRES_CONN = "dbname=iot user=postgres password=postgres host=postgres port=5432"
TOPIC = os.environ.get("KAFKA_TOPIC", "sensor.readings")
BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")

def consume_and_load(**context):
    """Load data from Kafka to Postgres (or BigQuery if configured)"""
    conn = psycopg2.connect(POSTGRES_CONN)
    cur = conn.cursor()
    
    consumer_config = {
        'bootstrap.servers': BROKER,
        'group.id': 'airflow-iot-consumer',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'session.timeout.ms': 30000,
        'heartbeat.interval.ms': 10000
    }
    consumer = Consumer(consumer_config)
    consumer.subscribe([TOPIC])
    
    count = 0
    timeout_count = 0
    max_timeouts = 30  # Stop after 30 consecutive timeouts (30 seconds)
    
    print(f"Starting Kafka consumer for topic: {TOPIC}")
    print(f"Broker: {BROKER}")
    
    try:
        while timeout_count < max_timeouts:
            msg = consumer.poll(1.0)  # Poll for 1 second
            
            if msg is None:
                timeout_count += 1
                if timeout_count % 10 == 0:
                    print(f"No messages received, timeout count: {timeout_count}/{max_timeouts}")
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print("Reached end of partition")
                    continue
                else:
                    print(f"Kafka error: {msg.error()}")
                    break
            
            # Reset timeout counter on successful message
            timeout_count = 0
            
            # Parse and insert message
            try:
                rec = json.loads(msg.value().decode('utf-8'))
                cur.execute(
                    """INSERT INTO raw_sensor_readings(event_time, sensor_id, temperature_c, humidity_pct, soil_moisture, payload)
                          VALUES (to_timestamp(%s), %s, %s, %s, %s, %s)""",
                    (rec["timestamp"], rec["sensor_id"], rec.get("temperature_c"), rec.get("humidity_pct"),
                     rec.get("soil_moisture"), json.dumps(rec)),
                )
                count += 1
                
                # Commit every 10 records
                if count % 10 == 0:
                    conn.commit()
                    print(f"Processed {count} records so far...")
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
                
    finally:
        consumer.close()
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Kafka consumer finished. Ingested {count} records total")

# Alternative BigQuery ingestion function (commented out)
# def consume_and_load_bigquery(**context):
#     """Load data from Kafka to BigQuery"""
#     from google.cloud import bigquery
#     
#     client = bigquery.Client()
#     table_id = f"{os.environ['GCP_PROJECT_ID']}.{os.environ['BQ_DATASET']}.raw_sensor_readings"
#     
#     consumer_config = {
#         'bootstrap.servers': BROKER,
#         'group.id': 'airflow-bigquery-consumer',
#         'auto.offset.reset': 'earliest',
#         'enable.auto.commit': False
#     }
#     consumer = Consumer(consumer_config)
#     consumer.subscribe([TOPIC])
#     
#     rows = []
#     for msg in consumer:
#         rec = msg.value
#         rows.append({
#             "event_time": datetime.fromtimestamp(rec["timestamp"]),
#             "sensor_id": rec["sensor_id"],
#             "temperature_c": rec.get("temperature_c"),
#             "humidity_pct": rec.get("humidity_pct"),
#             "soil_moisture": rec.get("soil_moisture"),
#             "payload": json.dumps(rec)
#         })
#     
#     if rows:
#         errors = client.insert_rows_json(table_id, rows)
#         if errors:
#             raise Exception(f"BigQuery insert errors: {errors}")
#         print(f"Ingested {len(rows)} records to BigQuery")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="iot_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    doc_md="Ingests Kafka topic into Postgres, then runs dbt models.",
):
    ingest = PythonOperator(task_id="consume_kafka_to_postgres", python_callable=consume_and_load)

    # dbt transformation tasks (temporarily disabled - dbt not installed)
    # dbt_deps = BashOperator(
    #     task_id="dbt_deps",
    #     bash_command="cd /opt/airflow/dbt && dbt deps --profiles-dir .",
    # )
    # 
    # dbt_run = BashOperator(
    #     task_id="dbt_run", 
    #     bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir .",
    # )
    # 
    # dbt_test = BashOperator(
    #     task_id="dbt_test",
    #     bash_command="cd /opt/airflow/dbt && dbt test --profiles-dir .",
    # )

    # Just run the ingestion for now
    # ingest >> dbt_deps >> dbt_run >> dbt_test
