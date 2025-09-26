-- Create BigQuery dataset and tables
CREATE SCHEMA IF NOT EXISTS `iot_pipeline`
OPTIONS(
  description="IoT sensor data pipeline dataset",
  location="US"
);

-- Raw sensor readings table
CREATE OR REPLACE TABLE `iot_pipeline.raw_sensor_readings` (
  event_time TIMESTAMP NOT NULL,
  sensor_id STRING NOT NULL,
  temperature_c FLOAT64,
  humidity_pct FLOAT64,
  soil_moisture FLOAT64,
  payload JSON NOT NULL
)
PARTITION BY DATE(event_time)
CLUSTER BY sensor_id
OPTIONS(
  description="Raw IoT sensor readings partitioned by date"
);





