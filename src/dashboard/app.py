#!/usr/bin/env python3
"""
Enhanced IoT Sensor Dashboard with ForecastWater API Integration.
Showcases the analytical service with real-time predictions.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Removed unused import: make_subplots
import requests
from datetime import datetime
import time

GA_MEASUREMENT_ID = "G-CN38644DDS"  # your GA4 Measurement ID

gtag = f"""
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
"""

st.components.v1.html(gtag, height=0)

# Configuration
DASHBOARD_MODE = os.getenv("DASHBOARD_MODE", "local").lower()
CLOUD_MODE = DASHBOARD_MODE == "cloud"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="IoT Agricultural Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.critical-alert {
    background-color: #ffebee;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f44336;
}
.warning-alert {
    background-color: #fff3e0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff9800;
}
.success-alert {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
}
/* Restrict main content width and center it */
section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)  # Cache for 30 seconds
def call_api(endpoint: str):
    """Make API call with error handling."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


@st.cache_data(ttl=60)  # Cache sensor data for 1 minute
def get_sensor_data():
    """Get sensor data from database (fallback if API fails)."""
    if CLOUD_MODE:
        PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        if not PROJECT_ID:
            st.error("Error: Set GCP_PROJECT_ID environment variable for cloud mode")
            return pd.DataFrame()

        from google.cloud import bigquery

        client = bigquery.Client(project=PROJECT_ID)
        BQ_DATASET = os.getenv("BQ_DATASET", "iot_pipeline")
        BQ_TABLE = os.getenv("BQ_TABLE", "raw_sensor_readings")

        query = f"""
        SELECT 
            sensor_id,
            event_time as timestamp,
            temperature_c as temperature,
            humidity_pct as humidity,
            soil_moisture
        FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        ORDER BY event_time DESC
        LIMIT 1000
        """
        return client.query(query).to_dataframe()
    else:
        from sqlalchemy import create_engine, text

        DB_USER = os.getenv("POSTGRES_USER", "postgres")
        DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
        DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
        DB_PORT = os.getenv("POSTGRES_PORT", "5433")
        DB_NAME = os.getenv("POSTGRES_DB", "iot")

        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        query = """
        SELECT 
            sensor_id,
            event_time as timestamp,
            temperature_c as temperature,
            humidity_pct as humidity,
            soil_moisture
        FROM raw_sensor_readings
        ORDER BY event_time DESC
        LIMIT 1000
        """
        return pd.read_sql(text(query), engine)


def main():
    # Header
    st.title("IoT Analytics Platform")
    st.markdown("**Real-time sensor monitoring with predictive watering analytics**")

    # Sidebar
    st.sidebar.header("System Status")

    # API Health Check
    health_data = call_api("/health")
    if health_data:
        api_status = health_data.get("status", "unknown")
        if api_status == "healthy":
            st.sidebar.success(f"API Status: {api_status.upper()}")
        else:
            st.sidebar.warning(f"API Status: {api_status.upper()}")
        db_status = health_data.get("database_connected", False)
        if db_status:
            st.sidebar.success("Database Status: Connected")
        else:
            st.sidebar.warning("Database Status: Disconnected")
    else:
        st.sidebar.error(" API Unavailable")

    # Mode indicator
    mode_text = "Cloud (BigQuery)" if CLOUD_MODE else "Local (PostgreSQL)"
    st.sidebar.info(f"**Mode**: {mode_text}")

    # Sensor Summary Table
    st.header("Sensor Status Summary")

    # Get all sensors from API
    sensors_data = call_api("/sensors")
    if sensors_data:
        summary_data = []
        # Cache prediction data to avoid multiple API calls
        prediction_cache = {}

        for sensor_meta in sensors_data:
            sensor_id = sensor_meta.get("sensor_id", "Unknown")

            # Get prediction for this sensor and cache it
            prediction_data = call_api(f"/sensors/{sensor_id}/predict")
            if prediction_data:
                prediction_cache[sensor_id] = prediction_data

            if prediction_data:
                status = prediction_data.get("status", "Unknown")
                predicted_date = prediction_data.get("predicted_watering_date")
                critical_date = prediction_data.get("critical_watering_date")
                current_moisture = prediction_data.get("current_moisture", 0)
                confidence_score = prediction_data.get("confidence_score", 0.0)

                # Determine watering date
                if "CRITICAL" in status:
                    watering_date = "TODAY (Critical)"
                elif critical_date:
                    try:
                        crit_dt = datetime.fromisoformat(
                            critical_date.replace("Z", "+00:00")
                        )
                        watering_date = crit_dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        watering_date = "Soon"
                elif predicted_date:
                    try:
                        pred_dt = datetime.fromisoformat(
                            predicted_date.replace("Z", "+00:00")
                        )
                        watering_date = pred_dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        watering_date = "Soon"
                else:
                    watering_date = "Not needed"

                summary_data.append(
                    {
                        "Sensor ID": sensor_id,
                        "Current Moisture": f"{current_moisture:.3f}",
                        "Status": status,
                        "ML Confidence": f"{confidence_score:.0%}",
                        "Projected Watering Date": watering_date,
                    }
                )
            else:
                summary_data.append(
                    {
                        "Sensor ID": sensor_id,
                        "Current Moisture": "N/A",
                        "Status": "API Error",
                        "ML Confidence": "N/A",
                        "Projected Watering Date": "N/A",
                    }
                )
        if summary_data:
            df_summary = pd.DataFrame(summary_data)

            # Style the table based on status
            def style_status(val):
                if "CRITICAL" in val:
                    return (
                        "background-color: #ffebee; color: #d32f2f; font-weight: bold"
                    )
                elif "WARNING" in val:
                    return (
                        "background-color: #fff3e0; color: #f57c00; font-weight: bold"
                    )
                elif "OK" in val:
                    return (
                        "background-color: #e8f5e8; color: #388e3c; font-weight: bold"
                    )
                return ""

            def style_watering_date(val):
                if "TODAY" in val or "Critical" in val:
                    return (
                        "background-color: #ffebee; color: #d32f2f; font-weight: bold"
                    )
                return ""

            styled_df = df_summary.style.applymap(
                style_status, subset=["Status"]
            ).applymap(style_watering_date, subset=["Projected Watering Date"])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Historical Data Visualization (now with prediction cache available)
    st.header("Historical Data & Trends")

    # Get raw sensor data for charts
    try:
        df_raw = get_sensor_data()

        if not df_raw.empty:
            # Sort data chronologically for proper time series visualization
            df_moisture = df_raw.sort_values("timestamp")

            # Soil moisture chart with predictions overlay
            fig_moisture = px.line(
                df_moisture,
                x="timestamp",
                y="soil_moisture",
                color="sensor_id",
                title="Soil Moisture Levels Over Time",
                labels={"soil_moisture": "Soil Moisture", "timestamp": "Time"},
            )

            # Add predicted decay curves for each sensor using cached data
            if prediction_cache:
                # Get unique sensor colors from the existing traces
                sensor_colors = {}
                for trace in fig_moisture.data:
                    sensor_id = trace.name
                    sensor_colors[sensor_id] = trace.line.color

                # Add decay curves for each sensor
                for sensor_id, prediction_data in prediction_cache.items():
                    decay_curve_data = prediction_data.get("predicted_decay_curve", {})
                    if decay_curve_data:
                        # Extract timestamps and moisture predictions
                        pred_timestamps = []
                        pred_moisture = []
                        for timestamp, moisture in decay_curve_data:
                            if timestamp and moisture is not None:
                                if moisture < 0.2:
                                    break
                                else:
                                    try:
                                        # Parse timestamp
                                        if isinstance(timestamp, str):
                                            timestamp_formatted = (
                                                datetime.fromisoformat(
                                                    timestamp.replace("Z", "+00:00")
                                                )
                                            )
                                        else:
                                            timestamp_formatted = timestamp

                                        pred_timestamps.append(timestamp_formatted)
                                        pred_moisture.append(moisture)
                                    except Exception:
                                        continue

                        # Add prediction curve if we have data
                        if pred_timestamps and pred_moisture:
                            # Get sensor color (use default if not found)
                            color = sensor_colors.get(sensor_id, "#1f77b4")

                            fig_moisture.add_trace(
                                go.Scatter(
                                    x=pred_timestamps,
                                    y=pred_moisture,
                                    mode="lines",
                                    name=f"{sensor_id} (Predicted)",
                                    line=dict(color=color, width=3, dash="dot"),
                                    opacity=0.7,
                                    hovertemplate="<b>%{fullData.name}</b><br>"
                                    + "Time: %{x}<br>"
                                    + "Predicted Moisture: %{y:.1%}<br>"
                                    + "<extra></extra>",
                                )
                            )

            # Add threshold lines
            fig_moisture.add_hline(
                y=0.2,
                line_dash="dash",
                line_color="red",
                annotation_text="Critical Threshold (0.2)",
            )
            fig_moisture.add_hline(
                y=0.3,
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning Threshold (0.3)",
            )

            # Update layout for better visualization
            fig_moisture.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig_moisture, use_container_width=True)

        else:
            st.warning("No sensor data available")

    except Exception as e:
        st.error(f"Error loading sensor data: {e}")

    # Get sensors from API
    sensors_data = call_api("/sensors")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Technical Stack**: FastAPI • Streamlit • PostgreSQL/BigQuery • Plotly • Docker"
    )
    st.markdown(f"**API Endpoint**: `{API_BASE_URL}`")

    time.sleep(3600)
    st.rerun()


if __name__ == "__main__":
    main()
