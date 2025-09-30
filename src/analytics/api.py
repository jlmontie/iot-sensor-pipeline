#!/usr/bin/env python3
"""
FastAPI service for ForecastWater analytics.
Exposes the analytical tool as a REST API.
"""

import os
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import logging

from fastapi import FastAPI, HTTPException
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
    redoc_url="/redoc",
)

# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests and responses
class PredictionRequest(BaseModel):
    """Request model for arbitrary sensor data prediction."""

    moisture_readings: List[float]
    timestamps: List[str]
    sensor_id: Optional[str] = "external-sensor"


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
    predicted_decay_curve: Optional[List[Tuple[datetime, float]]]
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
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        sensor_data = load_sensor_data(limit=1)
        database_connected = not sensor_data.empty
        sensors_available = (
            len(sensor_data["sensor_id"].unique()) if database_connected else 0
        )

        return HealthCheck(
            status="healthy" if database_connected else "degraded",
            timestamp=datetime.now(),
            database_connected=database_connected,
            sensors_available=sensors_available,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.now(),
            database_connected=False,
            sensors_available=0,
        )


@app.get("/sensors", response_model=List[SensorInfo])
async def get_sensors():
    """Get information about all available sensors."""
    try:
        sensor_data = load_sensor_data()

        if sensor_data.empty:
            return []

        sensors_info = []
        for sensor_id in sensor_data["sensor_id"].unique():
            sensor_df = sensor_data[sensor_data["sensor_id"] == sensor_id]

            sensors_info.append(
                SensorInfo(
                    sensor_id=sensor_id,
                    total_readings=len(sensor_df),
                    date_range_start=sensor_df["event_time"].min(),
                    date_range_end=sensor_df["event_time"].max(),
                    moisture_range_min=sensor_df["soil_moisture"].min(),
                    moisture_range_max=sensor_df["soil_moisture"].max(),
                )
            )

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

        if sensor_id not in sensor_data["sensor_id"].values:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")

        # Initialize forecaster and get analysis
        forecaster = SmartWateringPredictor()

        # Prepare data for SmartWateringPredictor
        sensor_subset = sensor_data[sensor_data["sensor_id"] == sensor_id].copy()
        sensor_subset = sensor_subset.rename(
            columns={"event_time": "timestamp", "soil_moisture": "moisture"}
        )

        if sensor_subset.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for sensor {sensor_id}"
            )

        # Get analysis results
        analysis = forecaster.analyze(sensor_subset)
        prediction = analysis["next_watering_prediction"]
        health = analysis["plant_health_metrics"]

        # Extract status from prediction urgency
        urgency_to_status = {
            "CRITICAL": "CRITICAL",
            "HIGH": "WARNING",
            "MEDIUM": "OK",
            "LOW": "OK",
            "NONE": "OK",
        }
        status = urgency_to_status.get(prediction["urgency"], "OK")

        return SensorStatus(
            sensor_id=sensor_id,
            current_moisture=health["current_moisture"],
            status=status,
            last_reading=sensor_subset["timestamp"].iloc[
                0
            ],  # Most recent due to DESC order
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

        if sensor_id not in sensor_data["sensor_id"].values:
            raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")

        # Initialize smart forecaster
        forecaster = SmartWateringPredictor()

        # Prepare data for SmartWateringPredictor
        sensor_subset = sensor_data[sensor_data["sensor_id"] == sensor_id].copy()
        sensor_subset = sensor_subset.rename(
            columns={"event_time": "timestamp", "soil_moisture": "moisture"}
        )

        # Get comprehensive analysis - this does all the work!
        analysis = forecaster.analyze(sensor_subset)

        # Extract results from analysis
        prediction = analysis["next_watering_prediction"]
        health = analysis["plant_health_metrics"]
        recommendations = analysis["recommendations"]

        # Map urgency to API status format
        urgency_to_status = {
            "CRITICAL": "CRITICAL",
            "HIGH": "WARNING",
            "MEDIUM": "OK",
            "LOW": "OK",
            "NONE": "OK",
        }
        status = urgency_to_status.get(prediction["urgency"], "OK")

        # Extract prediction dates
        predicted_date = prediction.get("timestamp")
        critical_date = None

        # For critical/high urgency, set critical date appropriately
        if prediction["urgency"] == "CRITICAL":
            critical_date = predicted_date  # Same as predicted for critical
        elif prediction["urgency"] == "HIGH":
            # Critical date is soon after predicted date
            from datetime import timedelta

            critical_date = (
                predicted_date + timedelta(hours=12) if predicted_date else None
            )

        # Calculate drying rate from recent data for API compatibility
        recent_data = sensor_subset.tail(48)  # Last 48 hours
        drying_rate = 0.0
        if len(recent_data) >= 2:
            time_span_hours = (
                recent_data["timestamp"].max() - recent_data["timestamp"].min()
            ).total_seconds() / 3600
            if time_span_hours > 0:
                moisture_change = (
                    recent_data["moisture"].iloc[0] - recent_data["moisture"].iloc[-1]
                )
                drying_rate = max(0.0, moisture_change / time_span_hours)

        curve_data = analysis.get("moisture_decay_curve", []).get("curve_data", [])
        predicted_decay_curve = [
            (item["timestamp"], item["predicted_moisture"]) for item in curve_data
        ]
        return PredictionResult(
            sensor_id=sensor_id,
            current_moisture=health["current_moisture"],
            status=status,
            predicted_watering_date=predicted_date,
            critical_watering_date=critical_date,
            drying_rate_per_hour=drying_rate,
            model_accuracy=analysis["model_confidence"],
            samples_used=len(sensor_subset),
            analysis_timestamp=datetime.now(),
            confidence_score=prediction.get("confidence", analysis["model_confidence"]),
            predicted_decay_curve=predicted_decay_curve,
            health_metrics=health,
            recommendations=recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting watering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sensors/predict/batch", response_model=List[PredictionResult])
async def predict_all_sensors():
    """Get watering predictions for all sensors."""
    try:
        sensor_data = load_sensor_data()

        if sensor_data.empty:
            raise HTTPException(status_code=404, detail="No sensor data available")

        predictions = []
        for sensor_id in sensor_data["sensor_id"].unique():
            try:
                # Initialize smart forecaster
                forecaster = SmartWateringPredictor()

                # Prepare data for SmartWateringPredictor
                sensor_subset = sensor_data[
                    sensor_data["sensor_id"] == sensor_id
                ].copy()
                sensor_subset = sensor_subset.rename(
                    columns={"event_time": "timestamp", "soil_moisture": "moisture"}
                )

                # Get comprehensive analysis
                analysis = forecaster.analyze(sensor_subset)

                # Extract results from analysis
                prediction = analysis["next_watering_prediction"]
                health = analysis["plant_health_metrics"]
                recommendations = analysis["recommendations"]

                # Map urgency to API status format
                urgency_to_status = {
                    "CRITICAL": "CRITICAL",
                    "HIGH": "WARNING",
                    "MEDIUM": "OK",
                    "LOW": "OK",
                    "NONE": "OK",
                }
                status = urgency_to_status.get(prediction["urgency"], "OK")

                # Extract prediction dates
                predicted_date = prediction.get("timestamp")
                critical_date = None

                # For critical/high urgency, set critical date appropriately
                if prediction["urgency"] == "CRITICAL":
                    critical_date = predicted_date
                elif prediction["urgency"] == "HIGH":
                    from datetime import timedelta

                    critical_date = (
                        predicted_date + timedelta(hours=12) if predicted_date else None
                    )

                # Calculate drying rate for API compatibility
                recent_data = sensor_subset.tail(48)
                drying_rate = 0.0
                if len(recent_data) >= 2:
                    time_span_hours = (
                        recent_data["timestamp"].max() - recent_data["timestamp"].min()
                    ).total_seconds() / 3600
                    if time_span_hours > 0:
                        moisture_change = (
                            recent_data["moisture"].iloc[0]
                            - recent_data["moisture"].iloc[-1]
                        )
                        drying_rate = max(0.0, moisture_change / time_span_hours)

                predictions.append(
                    PredictionResult(
                        sensor_id=sensor_id,
                        current_moisture=health["current_moisture"],
                        status=status,
                        predicted_watering_date=predicted_date,
                        critical_watering_date=critical_date,
                        drying_rate_per_hour=drying_rate,
                        model_accuracy=analysis["model_confidence"],
                        samples_used=len(sensor_subset),
                        analysis_timestamp=datetime.now(),
                        confidence_score=prediction.get(
                            "confidence", analysis["model_confidence"]
                        ),
                        predicted_decay_curve=prediction.get(
                            "moisture_decay_curve", []
                        ),
                        health_metrics=health,
                        recommendations=recommendations,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to predict for sensor {sensor_id}: {e}")
                continue

        return predictions

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResult)
async def predict_arbitrary_data(request: PredictionRequest):
    """
    Predict watering needs for arbitrary sensor data.

    This endpoint allows external users to submit their own moisture readings
    and timestamps to get ML-powered watering predictions.

    Example request:
    ```json
    {
       "moisture_readings": [0.98, 0.90, 0.84, 0.82, 0.77, 0.65, 0.58, 0.52, 0.45, 0.38],
       "timestamps": [
         "2025-09-01T10:00:00",
         "2025-09-02T10:00:00",
         "2025-09-03T10:00:00",
         "2025-09-04T10:00:00",
         "2025-09-05T10:00:00",
         "2025-09-06T10:00:00",
         "2025-09-07T10:00:00",
         "2025-09-08T10:00:00",
         "2025-09-09T10:00:00",
         "2025-09-10T10:00:00"
       ],
       "sensor_id": "my-garden-sensor"
    }
    ```
    """
    try:
        # Validate input
        if len(request.moisture_readings) != len(request.timestamps):
            raise HTTPException(
                status_code=400,
                detail="Number of moisture readings must match number of timestamps",
            )

        if len(request.moisture_readings) < 10:
            raise HTTPException(
                status_code=400,
                detail="At least 10 data points required for accurate prediction",
            )

        # Validate moisture values
        for moisture in request.moisture_readings:
            if not (0.0 <= moisture <= 1.0):
                raise HTTPException(
                    status_code=400,
                    detail="Moisture readings must be between 0.0 and 1.0",
                )

        # Convert to DataFrame format expected by SmartWateringPredictor
        try:
            timestamps_parsed = [
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                for ts in request.timestamps
            ]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timestamp format. Use ISO format (e.g., '2025-09-30T10:00:00'): {e}",
            )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "timestamp": timestamps_parsed,
                "moisture": request.moisture_readings,
                "sensor_id": [request.sensor_id] * len(request.moisture_readings),
            }
        )

        # Sort by timestamp to ensure proper time series order
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Initialize smart forecaster
        forecaster = SmartWateringPredictor()

        # Get comprehensive analysis
        analysis = forecaster.analyze(df)

        # Extract results from analysis
        current_moisture = df["moisture"].iloc[-1]  # Most recent reading

        next_watering = analysis.get("next_watering_prediction", {})
        predicted_date = next_watering.get("timestamp")
        
        # Calculate critical date based on when moisture reaches critical threshold
        critical_date = None
        decay_curve = analysis.get("moisture_decay_curve", {}).get("curve_data", [])
        for point in decay_curve:
            if point.get("status") == "CRITICAL":
                critical_date = point.get("timestamp")
                break

        # Determine status based on current moisture
        if current_moisture <= forecaster.critical_threshold:
            status = "CRITICAL"
        elif current_moisture <= forecaster.watering_threshold:
            status = "WARNING"
        else:
            status = "OK"

        # Get health metrics and recommendations
        plant_health = analysis.get("plant_health_metrics", {})
        recommendations = analysis.get("recommendations", [])
        confidence_score = analysis.get("model_confidence", 0.5)

        # Extract required fields from analysis or provide defaults
        drying_rate_per_hour = next_watering.get("hours_from_now", 24) / 24.0  # Convert to rate per hour
        model_accuracy = confidence_score
        samples_used = len(df)
        analysis_timestamp = analysis.get("analysis_metadata", {}).get("timestamp", datetime.now())
        
        # Prepare health_metrics dict for response
        health_metrics = {
            "health_score": plant_health.get("health_score", 85),
            "average_moisture_7d": plant_health.get("average_moisture_7d", current_moisture),
            "days_since_last_watering": plant_health.get("days_since_last_watering", 1.0),
            "moisture_stability": plant_health.get("moisture_stability", 0.1),
        }

        # Get predicted decay curve
        predicted_decay_curve = analysis.get("moisture_decay_curve", {}).get(
            "curve_points", []
        )

        return PredictionResult(
            sensor_id=request.sensor_id,
            current_moisture=current_moisture,
            status=status,
            predicted_watering_date=predicted_date,
            critical_watering_date=critical_date,
            drying_rate_per_hour=drying_rate_per_hour,
            model_accuracy=model_accuracy,
            samples_used=samples_used,
            analysis_timestamp=analysis_timestamp,
            confidence_score=confidence_score,
            health_metrics=health_metrics,
            recommendations=recommendations,
            predicted_decay_curve=predicted_decay_curve,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting arbitrary data: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Get port from environment (for Cloud Run compatibility)
    port = int(os.getenv("PORT", 8000))

    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True, log_level="info")
