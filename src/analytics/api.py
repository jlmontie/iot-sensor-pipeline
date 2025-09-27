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

from analytics.simple_forecaster import SimpleForecastWater

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
        
        # Initialize forecaster
        forecaster = SimpleForecastWater()
        forecaster.prepare(sensor_data, sensor_id)
        
        # Get current status
        status = forecaster.get_status()
        current_moisture = forecaster.data['soil_moisture'].iloc[-1]
        last_reading = forecaster.data['event_time'].iloc[-1]
        
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
        
        # Initialize forecaster
        forecaster = SimpleForecastWater()
        forecaster.prepare(sensor_data, sensor_id)
        
        # Get predictions
        predicted_date, critical_date, analysis = forecaster.predict_watering_date()
        status = forecaster.get_status()
        
        return PredictionResult(
            sensor_id=sensor_id,
            current_moisture=analysis['current_moisture'],
            status=status,
            predicted_watering_date=predicted_date,
            critical_watering_date=critical_date,
            drying_rate_per_hour=analysis['drying_rate_per_hour'],
            model_accuracy=analysis['r2_score'],
            samples_used=analysis['samples_used'],
            analysis_timestamp=datetime.now()
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
                # Initialize forecaster
                forecaster = SimpleForecastWater()
                forecaster.prepare(sensor_data, sensor_id)
                
                # Get predictions
                predicted_date, critical_date, analysis = forecaster.predict_watering_date()
                status = forecaster.get_status()
                
                predictions.append(PredictionResult(
                    sensor_id=sensor_id,
                    current_moisture=analysis['current_moisture'],
                    status=status,
                    predicted_watering_date=predicted_date,
                    critical_watering_date=critical_date,
                    drying_rate_per_hour=analysis['drying_rate_per_hour'],
                    model_accuracy=analysis['r2_score'],
                    samples_used=analysis['samples_used'],
                    analysis_timestamp=datetime.now()
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
