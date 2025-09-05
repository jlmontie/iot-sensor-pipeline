SELECT
  id,
  event_time,
  sensor_id,
  CAST(temperature_c AS DOUBLE PRECISION) AS temperature_c,
  CAST(humidity_pct AS DOUBLE PRECISION) AS humidity_pct,
  CAST(soil_moisture AS DOUBLE PRECISION) AS soil_moisture
FROM raw_sensor_readings;
