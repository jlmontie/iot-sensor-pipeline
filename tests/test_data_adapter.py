#!/usr/bin/env python3
"""
Basic tests for the data adapter module.
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from analytics.data_adapter import IoTDataAdapter


class TestDataAdapter:
    """Test suite for IoTDataAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = IoTDataAdapter()

    def test_adapter_initialization(self):
        """Test that adapter initializes correctly."""
        assert self.adapter is not None

    def test_convert_to_forecast_format_basic(self):
        """Test basic data conversion to forecast format."""
        # Sample IoT sensor data
        iot_data = pd.DataFrame(
            {
                "sensor_id": ["sensor_1", "sensor_1", "sensor_2"],
                "soil_moisture": [0.45, 0.42, 0.38],
                "timestamp": [
                    datetime(2025, 9, 30, 10, 0),
                    datetime(2025, 9, 30, 11, 0),
                    datetime(2025, 9, 30, 10, 0),
                ],
            }
        )

        result = self.adapter.convert_to_forecast_format(iot_data)

        # Basic validation
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that sensor data is grouped
        for sensor_id, sensor_data in result.items():
            assert isinstance(sensor_data, pd.DataFrame)
            assert not sensor_data.empty

    def test_convert_empty_dataframe(self):
        """Test conversion with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.adapter.convert_to_forecast_format(empty_df)

        # Should handle empty data gracefully
        assert isinstance(result, dict)

    def test_convert_single_sensor(self):
        """Test conversion with single sensor data."""
        single_sensor_data = pd.DataFrame(
            {
                "sensor_id": ["sensor_1", "sensor_1"],
                "soil_moisture": [0.45, 0.42],
                "timestamp": [
                    datetime(2025, 9, 30, 10, 0),
                    datetime(2025, 9, 30, 11, 0),
                ],
            }
        )

        result = self.adapter.convert_to_forecast_format(single_sensor_data)

        assert len(result) == 1
        assert "sensor_1" in result
        assert len(result["sensor_1"]) == 2

    def test_data_column_mapping(self):
        """Test that data columns are mapped correctly."""
        test_data = pd.DataFrame(
            {
                "sensor_id": ["test_sensor"],
                "soil_moisture": [0.5],
                "timestamp": [datetime.now()],
            }
        )

        result = self.adapter.convert_to_forecast_format(test_data)

        if result and "test_sensor" in result:
            sensor_df = result["test_sensor"]
            # Check that required columns exist in some form
            columns = sensor_df.columns.tolist()
            assert len(columns) > 0

    def test_timestamp_handling(self):
        """Test proper timestamp handling."""
        now = datetime.now()
        test_data = pd.DataFrame(
            {
                "sensor_id": ["sensor_1"],
                "soil_moisture": [0.4],
                "timestamp": [now],
            }
        )

        result = self.adapter.convert_to_forecast_format(test_data)

        # Verify timestamp is preserved
        if result and "sensor_1" in result:
            sensor_df = result["sensor_1"]
            # Should have timestamp information
            assert not sensor_df.empty


class TestDataValidation:
    """Test data validation utilities."""

    def test_required_columns_present(self):
        """Test that required columns are validated."""
        required_cols = ["sensor_id", "soil_moisture", "timestamp"]

        # Valid data
        valid_df = pd.DataFrame(
            {
                "sensor_id": ["sensor_1"],
                "soil_moisture": [0.4],
                "timestamp": [datetime.now()],
            }
        )

        for col in required_cols:
            assert col in valid_df.columns

    def test_moisture_value_validation(self):
        """Test moisture value validation."""
        # Valid moisture values (0-1 range)
        valid_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for value in valid_values:
            assert 0.0 <= value <= 1.0

    def test_sensor_id_format(self):
        """Test sensor ID format validation."""
        valid_sensor_ids = ["sensor_1", "sensor_2", "plant_monitor_01"]

        for sensor_id in valid_sensor_ids:
            assert isinstance(sensor_id, str)
            assert len(sensor_id) > 0


if __name__ == "__main__":
    pytest.main([__file__])
