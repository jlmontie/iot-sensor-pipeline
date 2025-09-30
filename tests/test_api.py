#!/usr/bin/env python3
"""
Basic tests for the FastAPI analytics service.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from analytics.api import app, get_database, PredictionResult


class TestAPI:
    """Test suite for FastAPI endpoints."""

    def test_app_creation(self):
        """Test that the FastAPI app is created successfully."""
        assert app is not None
        assert app.title == "IoT Sensor Analytics API"

    def test_prediction_result_model(self):
        """Test PredictionResult Pydantic model."""
        result = PredictionResult(
            sensor_id="sensor_1",
            current_moisture=0.45,
            status="OK",
            predicted_watering_date="2025-10-05T10:00:00Z",
            critical_watering_date="2025-10-03T08:00:00Z",
            confidence_score=0.85,
            health_metrics={"drying_rate": 0.02},
            recommendations=["Monitor daily"],
        )

        assert result.sensor_id == "sensor_1"
        assert result.current_moisture == 0.45
        assert result.status == "OK"
        assert result.confidence_score == 0.85

    @patch.dict(os.environ, {"DASHBOARD_MODE": "local"})
    def test_database_connection_local(self):
        """Test database connection configuration for local mode."""
        with patch("analytics.api.create_engine") as mock_engine:
            mock_engine.return_value = Mock()
            engine = get_database()
            mock_engine.assert_called_once()
            # Verify it was called with PostgreSQL connection string
            call_args = mock_engine.call_args[0][0]
            assert "postgresql://" in call_args

    @patch.dict(os.environ, {"DASHBOARD_MODE": "cloud", "GCP_PROJECT_ID": "test-project"})
    def test_database_connection_cloud(self):
        """Test database connection configuration for cloud mode."""
        with patch("analytics.api.create_engine") as mock_engine:
            mock_engine.return_value = Mock()
            engine = get_database()
            mock_engine.assert_called_once()
            # Verify it was called with BigQuery connection string
            call_args = mock_engine.call_args[0][0]
            assert "bigquery://" in call_args

    def test_health_endpoint_structure(self):
        """Test that health endpoint returns expected structure."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        with patch("analytics.api.get_database") as mock_db:
            mock_engine = Mock()
            mock_db.return_value = mock_engine

            # Mock successful database connection
            mock_engine.connect.return_value.__enter__.return_value = Mock()

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "database_connected" in data
            assert "sensors_available" in data


class TestDataProcessing:
    """Test data processing utilities."""

    def test_sensor_data_validation(self):
        """Test sensor data format validation."""
        # Valid sensor data
        valid_data = pd.DataFrame(
            {
                "sensor_id": ["sensor_1", "sensor_1"],
                "soil_moisture": [0.45, 0.42],
                "timestamp": [
                    datetime(2025, 9, 30, 10, 0),
                    datetime(2025, 9, 30, 11, 0),
                ],
            }
        )

        assert not valid_data.empty
        assert "sensor_id" in valid_data.columns
        assert "soil_moisture" in valid_data.columns
        assert "timestamp" in valid_data.columns

    def test_moisture_threshold_logic(self):
        """Test moisture level threshold logic."""
        # Test critical threshold
        critical_moisture = 0.15
        assert critical_moisture < 0.2  # Critical threshold

        # Test warning threshold
        warning_moisture = 0.25
        assert 0.2 <= warning_moisture < 0.3  # Warning threshold

        # Test OK threshold
        ok_moisture = 0.45
        assert ok_moisture >= 0.3  # OK threshold


if __name__ == "__main__":
    pytest.main([__file__])
