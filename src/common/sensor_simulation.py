"""
Shared sensor data simulation functions for consistent data generation
across historical frontloading and real-time streaming.
"""

import math
import random


def jitter(value, noise_level=0.05):
    """Add realistic noise to sensor readings"""
    return value + random.gauss(0, noise_level * abs(value))


def generate_temperature(sensor_id, t_hour):
    """Generate temperature with daily and seasonal cycles"""
    base_temp = 20 + 5 * math.sin(2 * math.pi * t_hour / (24 * 30))  # Monthly cycle
    daily_variation = 3 * math.sin(2 * math.pi * t_hour / 24)
    sensor_offset = (hash(sensor_id) % 100) / 100
    return base_temp + daily_variation + sensor_offset


def generate_humidity(sensor_id, t_hour):
    """Generate humidity with daily and seasonal cycles"""
    base_humidity = 60 + 10 * math.sin(2 * math.pi * t_hour / (24 * 30))
    daily_variation = 5 * math.sin(2 * math.pi * t_hour / 24 + math.pi / 2)
    sensor_offset = (hash(sensor_id) % 100) / 200
    return base_humidity + daily_variation + sensor_offset


def generate_moisture(sensor_id, t_hour):
    """Generate soil moisture with realistic watering cycles using exponential decay"""
    # Watering cycle parameters (customized per sensor)
    sensor_hash = hash(sensor_id) % 1000
    period = 168 + (sensor_hash % 48)  # 7±2 day watering cycles
    tau = 60 + (sensor_hash % 40)  # Decay rate variation
    amplitude = 0.8 + (sensor_hash % 100) / 500  # Peak moisture level

    # Phase within watering cycle [0, period)
    phase = t_hour % period

    # Exponential decay from peak moisture after watering
    base_moisture = amplitude * math.exp(-phase / tau)

    # Add small daily variation (evaporation patterns)
    daily_variation = 0.05 * math.sin(2 * math.pi * t_hour / 24)

    # Sensor-specific offset for calibration differences
    sensor_offset = (sensor_hash % 50) / 1000

    moisture = base_moisture + daily_variation + sensor_offset

    # Ensure realistic bounds (soil never completely dry or oversaturated)
    return max(0.15, min(0.9, moisture))


def generate_sensor_reading(sensor_id, t_hour, add_noise=True):
    """Generate a complete sensor reading with optional noise"""
    temp = generate_temperature(sensor_id, t_hour)
    humidity = generate_humidity(sensor_id, t_hour)
    moisture = generate_moisture(sensor_id, t_hour)
    
    if add_noise:
        temp = jitter(temp, 0.1)
        humidity = jitter(humidity, 0.05)
        moisture = jitter(moisture, 0.03)
    
    return {
        'temperature': round(temp, 2),
        'humidity': round(humidity, 1),
        'soil_moisture': round(moisture, 3)
    }
