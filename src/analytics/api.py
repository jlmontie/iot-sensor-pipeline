#!/usr/bin/env python3
"""
FastAPI service for ForecastWater analytics.
Exposes the analytical tool as a professional REST API.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, text

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.watering_predictor import SmartWateringPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ForecastWater Analytics API",
    description="Predictive watering analytics for IoT agricultural sensors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class SensorStatus(BaseModel):
    sensor_id: str
    current_moisture: float
    status: str
    last_reading: datetime
    
class PredictionResult(BaseModel):
    sensor_id: str
    current_moisture: float
    status: str
    predicted_watering_date: Optional[datetime]
    critical_watering_date: Optional[datetime]
    drying_rate_per_hour: float
    model_accuracy: float
    samples_used: int
    analysis_timestamp: datetime
    confidence_score: Optional[float]
    health_metrics: Optional[Dict[str, Any]]
    recommendations: Optional[List[str]]

class SensorInfo(BaseModel):
    sensor_id: str
    total_readings: int
    date_range_start: datetime
    date_range_end: datetime
    moisture_range_min: float
    moisture_range_max: float

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    database_connected: bool
    sensors_available: int

# Database connection
def get_database():
    """Get database connection based on environment."""
    # Check if we're in cloud mode
    dashboard_mode = os.getenv("DASHBOARD_MODE", "local").lower()
    
    if dashboard_mode == "cloud":
        # BigQuery connection (for cloud deployment)
        from google.cloud import bigquery
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise HTTPException(status_code=500, detail="GCP_PROJECT_ID not configured")
        return bigquery.Client(project=project_id)
    else:
        # PostgreSQL connection (for local development)
        DB_USER = os.getenv("POSTGRES_USER", "postgres")
        DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
        DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
        DB_PORT = os.getenv("POSTGRES_PORT", "5433")
        DB_NAME = os.getenv("POSTGRES_DB", "iot")
        
        return create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

def load_sensor_data(limit: int = 5000):
    """Load sensor data from database."""
    dashboard_mode = os.getenv("DASHBOARD_MODE", "local").lower()
    
    if dashboard_mode == "cloud":
        # BigQuery query
        client = get_database()
        dataset = os.getenv("BQ_DATASET", "iot_pipeline")
        table = os.getenv("BQ_TABLE", "raw_sensor_readings")
        
        query = f"""
        SELECT 
            sensor_id,
            event_time,
            temperature_c,
            humidity_pct,
            soil_moisture
        FROM `{client.project}.{dataset}.{table}`
        ORDER BY event_time DESC
        LIMIT {limit}
        """
        return client.query(query).to_dataframe()
    else:
        # PostgreSQL query
        engine = get_database()
        query = f"""
        SELECT 
            sensor_id,
            event_time,
            temperature_c,
            humidity_pct,
            soil_moisture
        FROM raw_sensor_readings
        ORDER BY event_time DESC
        LIMIT {limit}
        """
        return pd.read_sql(text(query), engine)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "ForecastWater Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        sensor_data = load_sensor_data(limit=1)
        database_connected = not sensor_data.empty
        sensors_available = len(sensor_data['sensor_id'].unique()) if database_connected else 0
        
        return HealthCheck(
            status="healthy" if database_connected else "degraded",
            timestamp=datetime.now(),
            database_connected=database_connected,
            sensors_available=sensors_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.now(),
            database_connected=False,
            sensors_available=0
        )

@app.get("/sensors", response_model=List[SensorInfo])
async def get_sensors():
    """Get information about all available sensors."""
    try:
        sensor_data = load_sensor_data()
        
        if sensor_data.empty:
            return []
        
        sensors_info = []
        for sensor_id in sensor_data['sensor_id'].unique():
            sensor_df = sensor_data[sensor_data['sensor_id'] == sensor_id]
            
            sensors_info.append(SensorInfo(
                sensor_id=sensor_id,
                total_readings=len(sensor_df),
                date_range_start=sensor_df['event_time'].min(),
                date_range_end=sensor_df['event_time'].max(),
                moisture_range_min=sensor_df['soil_moisture'].min(),
                moisture_range_max=sensor_df['soil_moisture'].max()
            ))
        
        return sensors_info
        
    except Exception as e:
        logger.error(f"Error getting sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensors/{sensor_id}/status", response_model=SensorStatus)
async def get_sensor_status(sensor_id: str):
    """Get current status for a specific sensor."""
    try:
        sensor_data = load_sensor_data()
        
        if sensor_data.empty:
            raise HTTPException(status_code=404, detail="No sensor data available")
        
        if sensor_id not in sensor_data['sensor_id'].values:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        
        # Get sensor data subset
        sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id].copy()
        
        if sensor_subset.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sensor {sensor_id}")
        
        # Get current status
        current_moisture = sensor_subset['soil_moisture'].iloc[-1]
        last_reading = sensor_subset['event_time'].iloc[-1]
        
        # Determine status based on thresholds
        if current_moisture <= 0.2:  # critical_threshold
            status = "critical"
        elif current_moisture <= 0.3:  # watering_threshold
            status = "warning"
        else:
            status = "ok"
        
        return SensorStatus(
            sensor_id=sensor_id,
            current_moisture=current_moisture,
            status=status,
            last_reading=last_reading
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sensor status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensors/{sensor_id}/predict", response_model=PredictionResult)
async def predict_watering(sensor_id: str):
    """Get watering predictions for a specific sensor."""
    try:
        sensor_data = load_sensor_data()
        
        if sensor_data.empty:
            raise HTTPException(status_code=404, detail="No sensor data available")
        
        if sensor_id not in sensor_data['sensor_id'].values:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
        
        # Initialize smart forecaster
        forecaster = SmartWateringPredictor()
        
        # Prepare data for SmartWateringPredictor (expects 'timestamp' and 'moisture' columns)
        sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id].copy()
        sensor_subset = sensor_subset.rename(columns={
            'event_time': 'timestamp',
            'soil_moisture': 'moisture'
        })
        
        # Get comprehensive analysis
        analysis = forecaster.analyze(sensor_subset)
        
        # Extract key information
        current_moisture = sensor_subset['moisture'].iloc[-1]
        current_time = datetime.now()  # Use actual current time for predictions
        
        # Determine status (uppercase for presentation)
        if current_moisture <= forecaster.critical_threshold:
            status = "CRITICAL"
        elif current_moisture <= forecaster.watering_threshold:
            status = "WARNING"
        else:
            status = "OK"
        
        # Generate predictions using simplified but reliable logic
        from sklearn.linear_model import LinearRegression
        import numpy as np
        from datetime import timedelta
        
        # Get recent moisture trend (last 100 points or all available)
        recent_data = sensor_subset.tail(100).copy()
        recent_data['hours_from_start'] = (
            recent_data['timestamp'] - recent_data['timestamp'].iloc[0]
        ).dt.total_seconds() / 3600
        
        # Fit linear regression to predict moisture decay
        X = recent_data['hours_from_start'].values.reshape(-1, 1)
        y = recent_data['moisture'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate drying rate and confidence
        drying_rate = abs(model.coef_[0])  # moisture units per hour
        r2_score = model.score(X, y)
        confidence_score = max(0.1, min(0.99, r2_score))  # Convert R² to confidence
        
        # Predict future dates
        predicted_date = None
        critical_date = None
        
        if drying_rate > 0.0001:  # Only predict if there's measurable drying
            # Hours until watering threshold (0.3)
            hours_to_watering = max(0, (current_moisture - forecaster.watering_threshold) / drying_rate)
            # Hours until critical threshold (0.2)  
            hours_to_critical = max(0, (current_moisture - forecaster.critical_threshold) / drying_rate)
            
            # Ensure predictions are always in the future (minimum 1 hour from now)
            hours_to_watering = max(1, hours_to_watering)
            hours_to_critical = max(1, hours_to_critical)
            
            if hours_to_watering < 8760:  # Within a year
                predicted_date = current_time + timedelta(hours=hours_to_watering)
            
            if hours_to_critical < 8760:  # Within a year
                critical_date = current_time + timedelta(hours=hours_to_critical)
        else:
            # If no drying trend, estimate based on typical plant behavior
            # Most plants need watering every 3-7 days
            days_estimate = 5 - (current_moisture - 0.3) * 10  # Higher moisture = longer time
            days_estimate = max(1, min(14, days_estimate))  # Clamp between 1-14 days
            
            predicted_date = current_time + timedelta(days=days_estimate)
            critical_date = current_time + timedelta(days=days_estimate + 2)
            confidence_score = 0.6  # Lower confidence for estimates
        
        # Get health metrics and recommendations from analysis
        health_metrics = analysis.get('plant_health_metrics', {})
        recommendations = analysis.get('recommendations', [])
        
        return PredictionResult(
            sensor_id=sensor_id,
            current_moisture=current_moisture,
            status=status,
            predicted_watering_date=predicted_date,
            critical_watering_date=critical_date,
            drying_rate_per_hour=health_metrics.get('drying_rate_per_hour', 0.0),
            model_accuracy=confidence_score,
            samples_used=len(sensor_subset),
            analysis_timestamp=datetime.now(),
            confidence_score=confidence_score,
            health_metrics=health_metrics,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting watering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sensors/{sensor_id}/predict/batch", response_model=List[PredictionResult])
async def predict_all_sensors():
    """Get watering predictions for all sensors."""
    try:
        sensor_data = load_sensor_data()
        
        if sensor_data.empty:
            raise HTTPException(status_code=404, detail="No sensor data available")
        
        predictions = []
        for sensor_id in sensor_data['sensor_id'].unique():
            try:
                # Initialize smart forecaster
                forecaster = SmartWateringPredictor()
                
                # Prepare data for SmartWateringPredictor
                sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id].copy()
                sensor_subset = sensor_subset.rename(columns={
                    'event_time': 'timestamp',
                    'soil_moisture': 'moisture'
                })
                
                # Get comprehensive analysis
                analysis = forecaster.analyze(sensor_subset)
                
                # Extract key information
                current_moisture = sensor_subset['moisture'].iloc[-1]
                next_watering = analysis.get('next_watering_prediction', {})
                health_metrics = analysis.get('plant_health_metrics', {})
                recommendations = analysis.get('recommendations', [])
                
                # Determine status (uppercase for presentation)
                if current_moisture <= forecaster.critical_threshold:
                    status = "CRITICAL"
                elif current_moisture <= forecaster.watering_threshold:
                    status = "WARNING"
                else:
                    status = "OK"
                
                # Extract dates
                predicted_date = None
                critical_date = None
                
                if next_watering and 'predicted_date' in next_watering:
                    predicted_date = next_watering['predicted_date']
                
                if next_watering and 'critical_date' in next_watering:
                    critical_date = next_watering['critical_date']
                
                predictions.append(PredictionResult(
                    sensor_id=sensor_id,
                    current_moisture=current_moisture,
                    status=status,
                    predicted_watering_date=predicted_date,
                    critical_watering_date=critical_date,
                    drying_rate_per_hour=health_metrics.get('drying_rate_per_hour', 0.0),
                    model_accuracy=forecaster.confidence_score,
                    samples_used=len(sensor_subset),
                    analysis_timestamp=datetime.now(),
                    confidence_score=forecaster.confidence_score,
                    health_metrics=health_metrics,
                    recommendations=recommendations
                ))
                
            except Exception as e:
                logger.warning(f"Failed to predict for sensor {sensor_id}: {e}")
                continue
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (for Cloud Run compatibility)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
