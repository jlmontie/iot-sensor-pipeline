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
echo "Next steps:"
echo "1. Start data generator:    python3 src/generator/simulate_stream.py local"
echo "2. Launch dashboard:        streamlit run src/dashboard/app.py"
echo "3. Access Airflow:          http://localhost:8080 (airflow/airflow)"
echo "4. Access Kafka UI:         http://localhost:8086"
echo ""
