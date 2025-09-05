CREATE TABLE IF NOT EXISTS raw_sensor_readings (
  id SERIAL PRIMARY KEY,
  event_time TIMESTAMP NOT NULL,
  sensor_id TEXT NOT NULL,
  temperature_c NUMERIC,
  humidity_pct NUMERIC,
  soil_moisture NUMERIC,
  payload JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kafka_offsets (
  topic TEXT NOT NULL,
  partition INT NOT NULL,
  last_offset BIGINT NOT NULL,
  PRIMARY KEY (topic, partition)
);
