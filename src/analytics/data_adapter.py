#!/usr/bin/env python3
"""
Data adapter to convert IoT sensor data to ForecastWater format.

This module bridges the gap between the IoT sensor data format and the
format expected by the ForecastWater analytical tool.
"""

import pandas as pd
import logging

# Removed unused imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTDataAdapter:
    """Converts IoT sensor data to ForecastWater-compatible format."""

    def __init__(self):
        """Initialize the data adapter."""
        pass

    def convert_sensor_data(
        self, sensor_data: pd.DataFrame, sensor_id: str
    ) -> pd.DataFrame:
        """
        Convert IoT sensor data to ForecastWater format.

        Args:
            sensor_data: DataFrame with columns [event_time, sensor_id, soil_moisture, ...]
            sensor_id: Specific sensor ID to filter and process

        Returns:
            DataFrame with columns [created_at, moisture, moisture_range] for ForecastWater
        """
        logger.info(f"Converting sensor data for sensor {sensor_id}")

        # Filter for specific sensor
        sensor_df = sensor_data[sensor_data["sensor_id"] == sensor_id].copy()

        if sensor_df.empty:
            logger.warning(f"No data found for sensor {sensor_id}")
            return pd.DataFrame()

        # Convert to ForecastWater format
        forecast_data = pd.DataFrame(
            {
                "created_at": pd.to_datetime(sensor_df["event_time"]),
                "moisture": sensor_df["soil_moisture"],
                # ForecastWater expects moisture_range for normalization
                # We'll use a default range based on typical soil moisture values
                "moisture_range": 1.0,  # Normalized range (0.0 to 1.0)
            }
        )

        # Sort by timestamp
        forecast_data = forecast_data.sort_values("created_at").reset_index(drop=True)

        logger.info(f"Converted {len(forecast_data)} records for sensor {sensor_id}")
        logger.info(
            f"Date range: {forecast_data['created_at'].min()} to {forecast_data['created_at'].max()}"
        )
        logger.info(
            f"Moisture range: {forecast_data['moisture'].min():.3f} to {forecast_data['moisture'].max():.3f}"
        )

        return forecast_data

    def get_available_sensors(self, sensor_data: pd.DataFrame) -> list:
        """
        Get list of available sensor IDs from the data.

        Args:
            sensor_data: DataFrame with sensor data

        Returns:
            List of unique sensor IDs
        """
        sensors = sensor_data["sensor_id"].unique().tolist()
        logger.info(f"Found {len(sensors)} sensors: {sensors}")
        return sensors
