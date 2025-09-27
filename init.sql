-- Initialize database for IoT Analytics Service
-- This creates the table structure needed for the ForecastWater analytics

CREATE TABLE IF NOT EXISTS raw_sensor_readings (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) NOT NULL,
    event_time TIMESTAMP NOT NULL,
    temperature_c FLOAT NOT NULL,
    humidity_pct FLOAT NOT NULL,
    soil_moisture FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id ON raw_sensor_readings(sensor_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_event_time ON raw_sensor_readings(event_time);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_time ON raw_sensor_readings(sensor_id, event_time);

-- Create a view for analytics queries
CREATE OR REPLACE VIEW sensor_analytics AS
SELECT 
    sensor_id,
    event_time,
    temperature_c,
    humidity_pct,
    soil_moisture,
    -- Calculate moving averages for trend analysis
    AVG(soil_moisture) OVER (
        PARTITION BY sensor_id 
        ORDER BY event_time 
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) as moisture_24h_avg,
    -- Calculate rate of change
    LAG(soil_moisture, 1) OVER (
        PARTITION BY sensor_id 
        ORDER BY event_time
    ) as prev_moisture
FROM raw_sensor_readings
ORDER BY sensor_id, event_time;
