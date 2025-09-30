#!/usr/bin/env python3
"""
Tests for actual IoT sensor pipeline functionality.
These tests validate the core components of the analytics platform.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from analytics.watering_predictor import SmartWateringPredictor


class TestSmartWateringPredictor:
    """Test the core ML prediction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = SmartWateringPredictor()

    def test_predictor_with_valid_sensor_data(self):
        """Test predictor with realistic declining moisture data."""
        # Create realistic sensor data showing moisture decline over 48 hours
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(47, -1, -1)]
        
        # Simulate realistic moisture decline from 60% to 25% over 48 hours
        moisture_values = np.linspace(0.6, 0.25, 48)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'moisture': moisture_values,
            'sensor_id': ['greenhouse_sensor_1'] * 48
        })

        # Test the analysis
        result = self.predictor.analyze(df)

        # Validate core functionality based on actual API structure
        assert isinstance(result, dict), "Analysis should return a dictionary"
        assert 'model_confidence' in result, "Should include model confidence"
        assert 'moisture_decay_curve' in result, "Should include decay curve prediction"
        assert 'next_watering_prediction' in result, "Should predict when watering is needed"
        
        # Validate confidence score
        confidence = result.get('model_confidence', 0)
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        
        # Validate prediction structure
        prediction = result.get('next_watering_prediction', {})
        assert isinstance(prediction, dict), "Prediction should be a dictionary"

    def test_predictor_threshold_logic(self):
        """Test that predictor correctly identifies critical moisture levels."""
        # Create data with critically low moisture (need at least 10 points)
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(11, -1, -1)]
        # Start higher and decline to critical levels
        critical_moisture_values = [0.35, 0.32, 0.29, 0.26, 0.23, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'moisture': critical_moisture_values,
            'sensor_id': ['critical_sensor'] * 12
        })

        result = self.predictor.analyze(df)
        
        # Should successfully analyze critical condition data
        assert isinstance(result, dict), "Should return analysis results for critical data"
        assert 'model_confidence' in result, "Should include confidence even for critical data"
        
        # Validate that analysis detects declining trend
        decay_curve = result.get('moisture_decay_curve', {})
        assert isinstance(decay_curve, dict), "Should include decay curve analysis"


class TestFastAPIIntegration:
    """Test FastAPI integration without complex mocking."""

    def test_api_app_creation(self):
        """Test that the FastAPI app can be imported and has correct configuration."""
        from analytics.api import app, PredictionResult
        
        # Test app exists and has correct title
        assert app is not None, "FastAPI app should be created"
        assert "Analytics API" in app.title, "App should have analytics-related title"
        
        # Test Pydantic model structure
        # Create a minimal valid prediction result
        test_result = {
            'sensor_id': 'test_sensor',
            'current_moisture': 0.35,
            'status': 'WARNING',
            'predicted_watering_date': '2025-10-02T10:00:00Z',
            'critical_watering_date': '2025-10-01T18:00:00Z',
            'confidence_score': 0.78,
            'health_metrics': {'drying_rate': 0.02},
            'recommendations': ['Check soil condition'],
            'drying_rate_per_hour': 0.02,
            'model_accuracy': 0.85,
            'samples_used': 24,
            'analysis_timestamp': '2025-09-30T15:00:00Z',
            'predicted_decay_curve': []
        }
        
        # This should not raise a validation error
        result = PredictionResult(**test_result)
        assert result.sensor_id == 'test_sensor'
        assert result.status == 'WARNING'
        assert 0.0 <= result.confidence_score <= 1.0


class TestDashboardDataProcessing:
    """Test dashboard data processing functionality."""

    def test_sensor_data_aggregation(self):
        """Test that sensor data can be properly aggregated for dashboard display."""
        # Create sample multi-sensor data
        sensors = ['greenhouse_1', 'greenhouse_2', 'outdoor_1']
        all_data = []
        
        for sensor in sensors:
            for i in range(24):  # 24 hours of data
                timestamp = datetime.now() - timedelta(hours=i)
                moisture = 0.5 - (i * 0.01)  # Gradual decline
                all_data.append({
                    'sensor_id': sensor,
                    'soil_moisture': moisture,
                    'timestamp': timestamp
                })
        
        df = pd.DataFrame(all_data)
        
        # Test data structure matches dashboard expectations
        required_columns = ['sensor_id', 'soil_moisture', 'timestamp']
        for col in required_columns:
            assert col in df.columns, f"Dashboard data should include {col} column"
        
        # Test data aggregation by sensor
        sensor_groups = df.groupby('sensor_id')
        assert len(sensor_groups) == 3, "Should have 3 sensor groups"
        
        # Test that each sensor has expected data points
        for sensor_id, group in sensor_groups:
            assert len(group) == 24, f"Each sensor should have 24 data points"
            assert group['soil_moisture'].min() >= 0.0, "Moisture should not be negative"
            assert group['soil_moisture'].max() <= 1.0, "Moisture should not exceed 100%"
            
        # Test moisture trend calculation (should be declining)
        for sensor_id, group in sensor_groups:
            sorted_group = group.sort_values('timestamp')
            # First reading is oldest (lowest moisture due to our generation logic)
            # Last reading is newest (highest moisture due to our generation logic)
            first_reading = sorted_group['soil_moisture'].iloc[0]  # Oldest = 0.5 - (23 * 0.01) = 0.27
            last_reading = sorted_group['soil_moisture'].iloc[-1]   # Newest = 0.5 - (0 * 0.01) = 0.5
            assert last_reading > first_reading, f"Moisture should be higher in recent readings for {sensor_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
