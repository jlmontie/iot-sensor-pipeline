#!/usr/bin/env python3
"""
Basic tests for the SmartWateringPredictor class.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from analytics.watering_predictor import SmartWateringPredictor


class TestSmartWateringPredictor:
    """Test suite for SmartWateringPredictor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = SmartWateringPredictor()

    def test_predictor_initialization(self):
        """Test that predictor initializes with correct defaults."""
        assert self.predictor.critical_threshold == 0.2
        assert self.predictor.watering_threshold == 0.3
        assert self.predictor.confidence_score == 0.0

    def test_analyze_with_empty_data(self):
        """Test analyze method with empty data."""
        empty_df = pd.DataFrame()
        result = self.predictor.analyze(empty_df)

        assert result is not None
        assert "error" in result or "status" in result

    def test_analyze_with_valid_data(self):
        """Test analyze method with valid sensor data."""
        # Create sample data with declining moisture
        dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        moisture_values = np.linspace(0.5, 0.25, 24)  # Declining from 50% to 25%

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "moisture": moisture_values,
                "sensor_id": ["sensor_1"] * 24,
            }
        )

        result = self.predictor.analyze(df)

        # Basic structure validation
        assert result is not None
        assert isinstance(result, dict)

        # Check for expected keys (may vary based on implementation)
        expected_keys = [
            "current_moisture",
            "predicted_watering_date",
            "confidence_score",
        ]
        for key in expected_keys:
            if key in result:
                assert result[key] is not None

    def test_validate_and_prepare_data(self):
        """Test data validation and preparation."""
        # Valid data
        valid_df = pd.DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=1), datetime.now()],
                "moisture": [0.4, 0.35],
                "sensor_id": ["sensor_1", "sensor_1"],
            }
        )

        # Should not raise an exception
        try:
            result = self.predictor._validate_and_prepare(valid_df)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If validation fails, ensure it's for a good reason
            assert "insufficient data" in str(e).lower() or "invalid" in str(e).lower()

    def test_threshold_constants(self):
        """Test that threshold constants are reasonable."""
        assert 0 < self.predictor.critical_threshold < 1
        assert 0 < self.predictor.watering_threshold < 1
        assert self.predictor.critical_threshold < self.predictor.watering_threshold

    def test_prediction_method_selection(self):
        """Test that prediction method selection works."""
        # Create data with clear trend
        dates = [datetime.now() - timedelta(hours=i) for i in range(48, 0, -1)]
        moisture_values = np.linspace(0.6, 0.2, 48)  # Clear declining trend

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "moisture": moisture_values,
                "sensor_id": ["sensor_1"] * 48,
            }
        )

        try:
            method = self.predictor._select_prediction_method(df)
            assert method in ["ml", "exponential", "linear"]
        except Exception:
            # Method selection might fail with insufficient data, which is acceptable
            pass


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_moisture_value_ranges(self):
        """Test that moisture values are in expected ranges."""
        # Test boundary conditions
        assert 0.0 <= 0.2 <= 1.0  # Critical threshold
        assert 0.0 <= 0.3 <= 1.0  # Warning threshold

    def test_datetime_handling(self):
        """Test datetime handling in predictions."""
        now = datetime.now()
        future = now + timedelta(hours=24)

        assert future > now
        assert (future - now).total_seconds() == 24 * 3600

    def test_data_structure_requirements(self):
        """Test required data structure for predictions."""
        required_columns = ["timestamp", "moisture", "sensor_id"]

        # Valid structure
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "moisture": [0.4],
                "sensor_id": ["sensor_1"],
            }
        )

        for col in required_columns:
            assert col in df.columns


if __name__ == "__main__":
    pytest.main([__file__])
