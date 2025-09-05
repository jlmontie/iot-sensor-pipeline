import os, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
import numpy as np

st.set_page_config(page_title="IoT Sensor Dashboard", layout="wide")

db_user = os.getenv("POSTGRES_USER", "postgres")
db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "iot")

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

st.title("🌱 IoT Sensor Dashboard")
st.markdown("Real-time monitoring of agricultural IoT sensors")

# Sidebar filters
st.sidebar.header("Filters")
time_range = st.sidebar.selectbox(
    "Time Range", 
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"],
    index=2
)

time_filters = {
    "Last Hour": "1 hour",
    "Last 6 Hours": "6 hours", 
    "Last 24 Hours": "24 hours",
    "Last Week": "7 days"
}

# Load recent data
with engine.connect() as con:
    df = pd.read_sql(text(f"""
        SELECT sensor_id, event_time, temperature_c, humidity_pct, soil_moisture
        FROM raw_sensor_readings
        WHERE event_time > NOW() - INTERVAL '{time_filters[time_range]}'
        ORDER BY event_time DESC
        LIMIT 5000
    """), con)

if df.empty:
    st.warning("No data available. Make sure the data generator is running and Airflow pipeline has executed.")
    st.stop()

# Convert event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Anomaly detection - flag low soil moisture
df['is_anomaly'] = df['soil_moisture'] < 0.15

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sensors", df['sensor_id'].nunique())
with col2:
    st.metric("Total Readings", len(df))
with col3:
    anomaly_count = df['is_anomaly'].sum()
    st.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_count/len(df)*100:.1f}%")
with col4:
    latest_reading = df['event_time'].max()
    st.metric("Latest Reading", latest_reading.strftime("%H:%M:%S") if pd.notnull(latest_reading) else "N/A")

# Time series visualization
st.subheader("📊 Sensor Readings Over Time")

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Soil Moisture'),
    shared_xaxes=True,
    vertical_spacing=0.08
)

# Temperature plot
for sensor in df['sensor_id'].unique():
    sensor_data = df[df['sensor_id'] == sensor]
    fig.add_trace(
        go.Scatter(
            x=sensor_data['event_time'], 
            y=sensor_data['temperature_c'],
            name=f'{sensor} - Temp',
            line=dict(width=2),
            showlegend=False
        ), 
        row=1, col=1
    )

# Humidity plot
for sensor in df['sensor_id'].unique():
    sensor_data = df[df['sensor_id'] == sensor]
    fig.add_trace(
        go.Scatter(
            x=sensor_data['event_time'], 
            y=sensor_data['humidity_pct'],
            name=f'{sensor} - Humidity',
            line=dict(width=2),
            showlegend=False
        ), 
        row=2, col=1
    )

# Soil moisture plot with anomaly highlighting
for sensor in df['sensor_id'].unique():
    sensor_data = df[df['sensor_id'] == sensor]
    normal_data = sensor_data[~sensor_data['is_anomaly']]
    anomaly_data = sensor_data[sensor_data['is_anomaly']]
    
    # Normal readings
    fig.add_trace(
        go.Scatter(
            x=normal_data['event_time'], 
            y=normal_data['soil_moisture'],
            name=f'{sensor} - Normal',
            line=dict(width=2),
            showlegend=False
        ), 
        row=3, col=1
    )
    
    # Anomaly readings
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['event_time'], 
                y=anomaly_data['soil_moisture'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name=f'{sensor} - Anomaly',
                showlegend=False
            ), 
            row=3, col=1
        )

fig.update_layout(height=800, title_text="Sensor Readings Timeline")
fig.update_xaxes(title_text="Time", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# Current status table
st.subheader("📋 Current Sensor Status")
with engine.connect() as con:
    current_status = pd.read_sql(text(f"""
        WITH latest_readings AS (
            SELECT sensor_id,
                   temperature_c,
                   humidity_pct, 
                   soil_moisture,
                   event_time,
                   ROW_NUMBER() OVER (PARTITION BY sensor_id ORDER BY event_time DESC) as rn
            FROM raw_sensor_readings
            WHERE event_time > NOW() - INTERVAL '{time_filters[time_range]}'
        )
        SELECT sensor_id,
               temperature_c,
               humidity_pct,
               soil_moisture,
               event_time,
               CASE WHEN soil_moisture < 0.15 THEN 'ALERT' ELSE 'OK' END as status
        FROM latest_readings 
        WHERE rn = 1
        ORDER BY sensor_id
    """), con)

# Style the dataframe
def highlight_anomalies(row):
    if row['status'] == 'ALERT':
        return ['background-color: #ffebee'] * len(row)
    return [''] * len(row)

if not current_status.empty:
    styled_df = current_status.style.apply(highlight_anomalies, axis=1)
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("No current sensor readings available.")

# Summary statistics
st.subheader("📈 Summary Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("**Average Values by Sensor**")
    summary_stats = df.groupby('sensor_id').agg({
        'temperature_c': 'mean',
        'humidity_pct': 'mean', 
        'soil_moisture': 'mean'
    }).round(2)
    st.dataframe(summary_stats, use_container_width=True)

with col2:
    st.write("**Anomaly Count by Sensor**")
    anomaly_summary = df.groupby('sensor_id')['is_anomaly'].sum().reset_index()
    anomaly_summary.columns = ['sensor_id', 'anomaly_count']
    
    fig_anomaly = px.bar(
        anomaly_summary, 
        x='sensor_id', 
        y='anomaly_count',
        title="Anomalies Detected per Sensor",
        color='anomaly_count',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
