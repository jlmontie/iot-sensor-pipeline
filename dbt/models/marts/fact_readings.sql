WITH base AS (
  SELECT * FROM {{ ref('stg_sensor_readings') }}
)
SELECT
  sensor_id,
  date_trunc('hour', event_time) AS hour_bucket,
  AVG(temperature_c) AS avg_temp_c,
  AVG(humidity_pct) AS avg_humidity,
  AVG(soil_moisture) AS avg_moisture
FROM base
GROUP BY sensor_id, date_trunc('hour', event_time);
