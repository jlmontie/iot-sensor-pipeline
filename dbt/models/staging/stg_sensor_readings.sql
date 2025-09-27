SELECT
  id,
  event_time,
  sensor_id,
  temperature_c::FLOAT AS temperature_c,
  humidity_pct::FLOAT AS humidity_pct,
  soil_moisture::FLOAT AS soil_moisture
FROM raw_sensor_readings
