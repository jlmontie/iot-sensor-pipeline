import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Simple mode detection for Streamlit
# Use environment variables since Streamlit doesn't handle command-line args well
DASHBOARD_MODE = os.getenv('DASHBOARD_MODE', 'local').lower()
CLOUD_MODE = DASHBOARD_MODE == 'cloud'

# Configuration based on mode
if CLOUD_MODE:
    PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    if not PROJECT_ID:
        st.error("Error: Set GCP_PROJECT_ID environment variable for cloud mode")
        st.info("Run with: DASHBOARD_MODE=cloud GCP_PROJECT_ID=your-project streamlit run src/dashboard/app.py")
        st.stop()
    BQ_DATASET = os.getenv('BQ_DATASET', 'iot_demo_dev_pipeline')
    BQ_TABLE = os.getenv('BQ_TABLE', 'sensor_readings')
else:
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5433")
    DB_NAME = os.getenv("POSTGRES_DB", "iot")

st.set_page_config(page_title="IoT Sensor Dashboard", layout="wide")

# Database connection based on environment
@st.cache_resource
def init_database():
    if CLOUD_MODE:
        from google.cloud import bigquery
        return bigquery.Client(project=PROJECT_ID)
    else:
        from sqlalchemy import create_engine
        return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

client = init_database()

# Title based on environment
title = "IoT Sensor Dashboard"
if CLOUD_MODE:
    title += " (Cloud)"
    st.markdown("Real-time monitoring of agricultural IoT sensors powered by Google Cloud")
else:
    st.markdown("Real-time monitoring of agricultural IoT sensors")

st.title(title)

def fetch_data(query):
    """Fetch data from either BigQuery or PostgreSQL based on environment"""
    try:
        if CLOUD_MODE:
            return client.query(query).to_dataframe()
        else:
            from sqlalchemy import text
            return pd.read_sql(text(query), client)
    except Exception as e:
        if CLOUD_MODE and "db-dtypes" in str(e):
            st.error("Missing required package: db-dtypes")
            st.code("pip install db-dtypes>=1.0.0")
            st.info("This package is required for BigQuery data type handling")
        elif CLOUD_MODE and "not found" in str(e).lower():
            st.error("BigQuery table not found. Make sure data generator is running and has sent data.")
            st.info(f"Looking for: `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`")
        else:
            st.error(f"Database error: {e}")
        raise e

def get_sensor_data():
    """Get sensor data with environment-specific query"""
    if CLOUD_MODE:
        query = f"""
        SELECT 
            sensor_id,
            timestamp,
            temperature,
            humidity,
            soil_moisture
        FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY timestamp DESC
        LIMIT 1000
        """
    else:
        query = """
        SELECT 
            sensor_id,
            event_time as timestamp,
            temperature_c as temperature,
            humidity_pct as humidity,
            soil_moisture
        FROM raw_sensor_readings
        WHERE event_time >= NOW() - INTERVAL '24 hours'
        ORDER BY event_time DESC
        LIMIT 1000
        """
    
    return fetch_data(query)

def get_aggregated_data():
    """Get aggregated data with environment-specific query"""
    if CLOUD_MODE:
        query = f"""
        SELECT 
            sensor_id,
            TIMESTAMP_TRUNC(timestamp, HOUR) as hour,
            AVG(temperature) as avg_temperature,
            AVG(humidity) as avg_humidity,
            AVG(soil_moisture) as avg_soil_moisture,
            COUNT(*) as reading_count
        FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        GROUP BY sensor_id, TIMESTAMP_TRUNC(timestamp, HOUR)
        ORDER BY hour DESC
        """
    else:
        query = """
        SELECT 
            sensor_id,
            DATE_TRUNC('hour', event_time) as hour,
            AVG(temperature_c) as avg_temperature,
            AVG(humidity_pct) as avg_humidity,
            AVG(soil_moisture) as avg_soil_moisture,
            COUNT(*) as reading_count
        FROM raw_sensor_readings
        WHERE event_time >= NOW() - INTERVAL '24 hours'
        GROUP BY sensor_id, DATE_TRUNC('hour', event_time)
        ORDER BY hour DESC
        """
    
    return fetch_data(query)

# Main dashboard logic
try:
    # Fetch data
    df_raw = get_sensor_data()
    df_agg = get_aggregated_data()
    
    if df_raw.empty:
        st.warning("No data available. Make sure the data pipeline is running.")
        st.stop()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sensors = df_raw['sensor_id'].nunique()
        st.metric("Active Sensors", total_sensors)
    
    with col2:
        latest_temp = df_raw['temperature'].iloc[0] if not df_raw.empty else 0
        st.metric("Latest Temperature", f"{latest_temp:.1f}°C")
    
    with col3:
        latest_humidity = df_raw['humidity'].iloc[0] if not df_raw.empty else 0
        st.metric("Latest Humidity", f"{latest_humidity:.1f}%")
    
    with col4:
        latest_moisture = df_raw['soil_moisture'].iloc[0] if not df_raw.empty else 0
        st.metric("Latest Soil Moisture", f"{latest_moisture:.3f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Trends")
        if not df_agg.empty:
            fig_temp = px.line(df_agg, x='hour', y='avg_temperature', 
                              color='sensor_id', title="Average Temperature by Hour")
            st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        st.subheader("Humidity Trends")
        if not df_agg.empty:
            fig_humidity = px.line(df_agg, x='hour', y='avg_humidity',
                                  color='sensor_id', title="Average Humidity by Hour")
            st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Soil moisture with anomaly detection
    st.subheader("Soil Moisture Analysis")
    if not df_raw.empty:
        # Simple anomaly detection (values below 0.2 are concerning)
        # Create anomaly detection (boolean for alerts, numeric for visualization)
        df_raw['anomaly'] = df_raw['soil_moisture'] < 0.2
        df_raw['anomaly_size'] = df_raw['anomaly'].astype(int) * 10 + 5  # 15 for anomalies, 5 for normal
        
        fig_moisture = px.scatter(df_raw, x='timestamp', y='soil_moisture',
                                 color='sensor_id', size='anomaly_size',
                                 title="Soil Moisture Levels (Large points = Low moisture alerts)")
        st.plotly_chart(fig_moisture, use_container_width=True)
        
        # Alert for low moisture
        low_moisture = df_raw[df_raw['anomaly']]['sensor_id'].unique()
        if len(low_moisture) > 0:
            st.error(f"Low soil moisture alert for sensors: {', '.join(low_moisture)}")
    
    # Recent readings table
    st.subheader("Recent Readings")
    st.dataframe(df_raw.head(20), use_container_width=True)
    
    # Environment info
    st.sidebar.info(f"Running in {'Cloud' if CLOUD_MODE else 'Local'} mode")
    if CLOUD_MODE:
        st.sidebar.info(f"Dataset: {os.getenv('BQ_DATASET', 'iot_demo_dev_pipeline')}")
    
except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("Make sure your database/BigQuery is accessible and contains data.")
