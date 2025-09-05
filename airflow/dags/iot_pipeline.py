from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
# Uncomment for BigQuery integration:
# from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
# from google.cloud import bigquery
import json, os, time
import psycopg2
from kafka import KafkaConsumer

POSTGRES_CONN = "dbname=iot user=postgres password=postgres host=postgres port=5432"
TOPIC = os.environ.get("KAFKA_TOPIC", "sensor.readings")
BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")

def consume_and_load(**context):
    """Load data from Kafka to Postgres (or BigQuery if configured)"""
    conn = psycopg2.connect(POSTGRES_CONN)
    cur = conn.cursor()
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=10000,  # stop after idle
    )
    count = 0
    for msg in consumer:
        rec = msg.value
        cur.execute(
            \"""INSERT INTO raw_sensor_readings(event_time, sensor_id, temperature_c, humidity_pct, soil_moisture, payload)
                  VALUES (to_timestamp(%s), %s, %s, %s, %s, %s)\"""",
            (rec["timestamp"], rec["sensor_id"], rec.get("temperature_c"), rec.get("humidity_pct"),
             rec.get("soil_moisture"), json.dumps(rec)),
        )
        count += 1
    conn.commit()
    cur.close(); conn.close()
    print(f"Ingested {count} records")

# Alternative BigQuery ingestion function (commented out)
# def consume_and_load_bigquery(**context):
#     """Load data from Kafka to BigQuery"""
#     from google.cloud import bigquery
#     
#     client = bigquery.Client()
#     table_id = f"{os.environ['GCP_PROJECT_ID']}.{os.environ['BQ_DATASET']}.raw_sensor_readings"
#     
#     consumer = KafkaConsumer(
#         TOPIC,
#         bootstrap_servers=[BROKER],
#         value_deserializer=lambda m: json.loads(m.decode("utf-8")),
#         auto_offset_reset="earliest",
#         enable_auto_commit=False,
#         consumer_timeout_ms=10000,
#     )
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

default_args = {{
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}}

with DAG(
    dag_id="iot_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    doc_md="Ingests Kafka topic into Postgres, then runs dbt models.",
):
    ingest = PythonOperator(task_id="consume_kafka_to_postgres", python_callable=consume_and_load)

    # dbt transformation tasks
    dbt_deps = BashOperator(
        task_id="dbt_deps",
        bash_command="cd /opt/airflow/dags/../../dbt && dbt deps --profiles-dir .",
    )
    
    dbt_run = BashOperator(
        task_id="dbt_run", 
        bash_command="cd /opt/airflow/dags/../../dbt && dbt run --profiles-dir .",
    )
    
    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/airflow/dags/../../dbt && dbt test --profiles-dir .",
    )

    ingest >> dbt_deps >> dbt_run >> dbt_test
