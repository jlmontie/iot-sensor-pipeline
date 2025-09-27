{{
  config(
    materialized='table',
    partition_by={
      "field": "hour_bucket",
      "data_type": "timestamp",
      "granularity": "hour"
    },
    cluster_by=['sensor_id']
  )
}}

WITH base AS (
  SELECT * FROM {{ ref('stg_sensor_readings') }}
)
SELECT
  sensor_id,
  TIMESTAMP_TRUNC(event_time, HOUR) AS hour_bucket,
  AVG(temperature_c) AS avg_temp_c,
  MIN(temperature_c) AS min_temp_c,
  MAX(temperature_c) AS max_temp_c,
  AVG(humidity_pct) AS avg_humidity,
  MIN(humidity_pct) AS min_humidity,
  MAX(humidity_pct) AS max_humidity,
  AVG(soil_moisture) AS avg_moisture,
  MIN(soil_moisture) AS min_moisture,
  MAX(soil_moisture) AS max_moisture,
  COUNT(*) AS reading_count,
  -- Anomaly detection
  COUNTIF(soil_moisture < 0.15) AS anomaly_count
FROM base
GROUP BY sensor_id, TIMESTAMP_TRUNC(event_time, HOUR)





