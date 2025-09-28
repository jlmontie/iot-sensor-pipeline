#!/usr/bin/env python3
"""
Simplified forecaster that works directly with IoT sensor data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class SimpleForecastWater:
    """Simplified watering forecaster for IoT sensor data."""
    
    def __init__(self, critical_moisture_threshold: float = 0.2):
        """
        Initialize the forecaster.
        
        Args:
            critical_moisture_threshold: Moisture level below which watering is critical
        """
        self.critical_threshold = critical_moisture_threshold
        self.warning_threshold = critical_moisture_threshold + 0.1  # 10% buffer
        self.data = None
        
    def prepare(self, sensor_data: pd.DataFrame, sensor_id: str) -> None:
        """
        Prepare data for a specific sensor.
        
        Args:
            sensor_data: DataFrame with columns [event_time, sensor_id, soil_moisture, ...]
            sensor_id: Specific sensor to analyze
        """
        # Filter for specific sensor
        self.data = sensor_data[sensor_data['sensor_id'] == sensor_id].copy()
        
        if self.data.empty:
            raise ValueError(f"No data found for sensor {sensor_id}")
        
        # Sort by time
        self.data = self.data.sort_values('event_time').reset_index(drop=True)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data['event_time']):
            self.data['event_time'] = pd.to_datetime(self.data['event_time'])
        
        logger.info(f"Prepared {len(self.data)} records for sensor {sensor_id}")
        logger.info(f"Date range: {self.data['event_time'].min()} to {self.data['event_time'].max()}")
        logger.info(f"Moisture range: {self.data['soil_moisture'].min():.3f} to {self.data['soil_moisture'].max():.3f}")
    
    def predict_watering_date(self) -> Tuple[Optional[datetime], Optional[datetime], dict]:
        """
        Predict when watering will be needed.
        
        Returns:
            Tuple of (predicted_date, critical_date, analysis_info)
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data prepared. Call prepare() first.")
        
        # Get recent data (last 7 days) for trend analysis
        recent_cutoff = self.data['event_time'].max() - timedelta(days=7)
        recent_data = self.data[self.data['event_time'] >= recent_cutoff].copy()
        
        if len(recent_data) < 10:
            logger.warning("Insufficient recent data for reliable prediction")
            recent_data = self.data.tail(50)  # Use last 50 points
        
        # Current moisture level
        current_moisture = recent_data['soil_moisture'].iloc[-1]
        current_time = recent_data['event_time'].iloc[-1]
        
        # Calculate drying rate using linear regression
        recent_data['hours_from_start'] = (
            recent_data['event_time'] - recent_data['event_time'].iloc[0]
        ).dt.total_seconds() / 3600
        
        # Fit linear regression to recent moisture trend
        X = recent_data['hours_from_start'].values.reshape(-1, 1)
        y = recent_data['soil_moisture'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Drying rate (slope) - negative means moisture is decreasing
        drying_rate = model.coef_[0]  # moisture units per hour
        
        analysis_info = {
            'current_moisture': current_moisture,
            'current_time': current_time,
            'drying_rate_per_hour': drying_rate,
            'samples_used': len(recent_data),
            'r2_score': model.score(X, y),
            'critical_threshold': self.critical_threshold,
            'warning_threshold': self.warning_threshold
        }
        
        # Predict when thresholds will be reached
        predicted_date = None
        critical_date = None
        
        hours_to_warning = (current_moisture - self.warning_threshold) / abs(drying_rate)
        predicted_date = current_time + timedelta(hours=hours_to_warning)
        logger.info(f"Current moisture < warning threshold. \nHours to warning: {hours_to_warning}\nPredicted date: {predicted_date}\n")
            
        hours_to_critical = (current_moisture - self.critical_threshold) / abs(drying_rate)
        critical_date = current_time + timedelta(hours=hours_to_critical)
        logger.info(f"Current moisture < critical threshold. \nHours to critical: {hours_to_critical}\nCritical date: {critical_date}\n")
        
        logger.info(f"Analysis info: {analysis_info}")
        logger.info(f"current_moisture < self.warning_threshold: {current_moisture < self.warning_threshold}")
        logger.info(f"Prediction complete for recent {len(recent_data)} samples")
        logger.info(f"Current moisture: {current_moisture:.3f}")
        logger.info(f"Drying rate: {drying_rate:.6f} per hour")
        logger.info(f"Predicted watering date: {predicted_date}")
        logger.info(f"Critical watering date: {critical_date}")
        
        return predicted_date, critical_date, analysis_info
    
    def get_status(self) -> str:
        """Get current watering status."""
        if self.data is None or self.data.empty:
            return "No data"
        
        current_moisture = self.data['soil_moisture'].iloc[-1]
        
        if current_moisture <= self.critical_threshold:
            return "CRITICAL - Water immediately"
        elif current_moisture <= self.warning_threshold:
            return "WARNING - Water soon"
        else:
            return "OK - No watering needed"
