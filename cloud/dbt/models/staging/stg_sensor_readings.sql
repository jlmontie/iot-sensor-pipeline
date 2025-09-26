{{
  config(
    materialized='view',
    partition_by={
      "field": "event_time",
      "data_type": "timestamp",
      "granularity": "day"
    }
  )
}}

SELECT
  event_time,
  sensor_id,
  CAST(temperature_c AS FLOAT64) AS temperature_c,
  CAST(humidity_pct AS FLOAT64) AS humidity_pct,
  CAST(soil_moisture AS FLOAT64) AS soil_moisture,
  -- Extract additional fields from JSON payload if needed
  JSON_EXTRACT_SCALAR(payload, '$.timestamp') as original_timestamp
FROM {{ source('iot_pipeline', 'raw_sensor_readings') }}
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)  -- Cost optimization





