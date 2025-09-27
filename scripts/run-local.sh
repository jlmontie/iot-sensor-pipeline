#!/bin/bash
# Local development deployment script

set -e

echo "Starting local IoT pipeline..."

# Set local environment variables for compatibility
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DB=iot
export KAFKA_BROKER=localhost:9092
export KAFKA_TOPIC=sensor.readings

# Start services
echo "Starting Docker services..."
docker compose up -d

echo "Waiting for services to be ready..."
sleep 30

# Install dependencies
echo "Installing Python dependencies..."
pip install -r src/requirements.txt

echo "Local environment ready!"
echo ""
echo "Starting data generator..."
python3 src/generator/simulate_stream.py local &
DATA_GEN_PID=$!

echo "Frontloading dashboard with 1000 historical data points..."
python3 scripts/frontload-dashboard-data.py &
FRONTLOAD_PID=$!

echo "Waiting for services to be ready..."
sleep 30

echo ""
echo "Local environment ready!"
echo ""
echo "Services:"
echo "1. Data generator:          Running (PID: $DATA_GEN_PID)"
echo "2. Dashboard data:          Running (PID: $FRONTLOAD_PID) - 1000+ points loaded"
echo "3. Dashboard:               streamlit run src/dashboard/app.py"
echo "4. Airflow:                 http://localhost:8080 (airflow/airflow)"
echo "5. Kafka UI:                http://localhost:8086"
echo ""
echo "To stop data processes:"
echo "  kill $DATA_GEN_PID $FRONTLOAD_PID"
echo ""
echo "Dashboard will show rich historical data immediately!"
echo ""
