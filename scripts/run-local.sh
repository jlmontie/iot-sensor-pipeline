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

# Clean up any existing containers and volumes for fresh start
echo "Cleaning up existing Docker resources..."
docker compose down -v 2>/dev/null || true

# Start services
echo "Starting Docker services..."
docker compose up -d

echo "Waiting for services to be ready..."
sleep 30

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install -r src/requirements.txt

echo "Local environment ready!"
echo ""
echo "Frontloading dashboard with 1000 historical data points..."
python3 scripts/frontload-dashboard-data.py

echo "Waiting for PostgreSQL to initialize..."
sleep 10

echo ""
echo "=== LOCAL ENVIRONMENT READY ==="
echo ""
echo "Services:"
echo "1. PostgreSQL Database:     localhost:5433 (postgres/postgres)"
echo "2. Real-time data generator: Running (PID: $DATA_GEN_PID)"
echo "3. Dashboard:               streamlit run src/dashboard/app.py"
echo ""
echo "To stop everything:"
echo "  ./scripts/stop-local.sh"
echo ""
echo "Dashboard shows 6 weeks of historical data"
echo ""
