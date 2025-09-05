# IoT Sensor Data Pipeline (Portfolio Project)

End-to-end data engineering project that simulates streaming IoT sensor data and runs it through a modern stack:

- **Ingestion/Streaming:** Kafka (via docker-compose)
- **Orchestration:** Apache Airflow
- **Warehouse:** Postgres (local dev) — swap to **BigQuery** with provided hooks
- **Transformations:** dbt (staging → marts)
- **Analytics App:** Streamlit dashboard
- **Infra:** Docker Compose, .env configuration

## 🎯 Business Value & Impact

**Problem Solved**: Traditional agricultural monitoring relies on manual inspections, leading to:
- Late detection of crop stress (soil moisture drops)
- Inefficient irrigation scheduling 
- Reactive rather than proactive farm management
- Higher operational costs and crop losses

**Solution Delivered**: Real-time IoT sensor monitoring with automated anomaly detection that enables:
- **Proactive Maintenance**: Early warning system for irrigation issues
- **Reduced Truck Rolls**: Remote monitoring eliminates unnecessary field visits  
- **Optimized Resource Usage**: Data-driven irrigation and climate control
- **Predictive Analytics**: Historical trends inform future planting decisions

**Technical Excellence**: This project demonstrates production-ready data engineering skills:
- **Streaming Data Processing**: Kafka for real-time sensor ingestion
- **Workflow Orchestration**: Airflow for reliable, scheduled pipeline execution
- **Data Quality Assurance**: Comprehensive dbt testing and validation
- **Dimensional Modeling**: Clean staging → marts architecture
- **Interactive Analytics**: Real-time dashboard with anomaly highlighting

## Quickstart

1) **Clone** this repo and copy `.env.example` to `.env`, then adjust values if desired.
   ```bash
   cp .env.example .env
   ```
2) **Start services**:
```bash
docker compose up -d
```
This boots: Zookeeper, Kafka, Postgres, Airflow Webserver/Scheduler, and a lightweight Kafka UI.

3) **Init Postgres schema** (first run only):
```bash
docker compose exec postgres psql -U postgres -d iot -f /docker-entrypoint-initdb.d/init.sql
```

4) **Start the simulator** (from your host):
```bash
pip install -r generator/requirements.txt
python generator/simulate_stream.py
```
It sends JSON messages to Kafka topic `sensor.readings`.

5) **Airflow UI** at http://localhost:8080 (user: `airflow`, pass: `airflow`).
   - Turn on the DAG **iot_pipeline** to:
     - Read from Kafka and land raw JSON into Postgres (`raw_sensor_readings`)
     - Run dbt models to build `stg_` and `dim_/fact_` tables

6) **Run the dashboard** (from host, new terminal):
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Dashboard at http://localhost:8501.

## BigQuery Option
If you prefer BigQuery over Postgres:
- Create a service account key JSON and mount it in `docker-compose.yml` (see commented lines).
- Swap the Airflow task to use `BigQueryInsertJobOperator` (comment included in DAG).

## Repo Structure
```
airflow/
  dags/iot_pipeline.py
dbt/
  models/staging/*.sql
  models/marts/*.sql
generator/
  simulate_stream.py
dashboard/
  app.py
sql/
  init.sql
diagrams/
  architecture.mmd
```

## 🎤 Interview Talking Points

**Architecture Decisions**:
- **Kafka vs Batch**: Real-time streaming enables immediate anomaly detection vs waiting hours for batch processing
- **Airflow Orchestration**: Reliable scheduling, retry logic, and dependency management for production workloads
- **dbt for Transformations**: Version-controlled, testable SQL transformations with lineage tracking

**Data Quality & Reliability**:
- **Comprehensive Testing**: Range validations, uniqueness constraints, and referential integrity checks
- **Staging Layer**: Clean, typed data before business logic to catch issues early
- **Anomaly Detection**: Automated flagging of soil moisture drops for proactive maintenance

**Scalability Considerations**:
- **Incremental Models**: dbt incremental strategies for handling growing datasets
- **Partitioning**: Time-based partitioning for query performance optimization  
- **Cloud Migration**: BigQuery hooks provided for seamless cloud scaling

**Production Readiness**:
- **Monitoring**: Airflow SLAs and task-level metrics for operational visibility
- **Error Handling**: Retry policies and dead letter queues for fault tolerance
- **Future Extensions**: ML model retraining pipelines, real-time alerting systems

---
MIT License © You
