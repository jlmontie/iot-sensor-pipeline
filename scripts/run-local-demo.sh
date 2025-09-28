#!/bin/bash
# Run the complete analytics demo

set -e

echo "IoT Agricultural Analytics Platform - Demo"
echo "=========================================="
echo ""

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if database is running
if ! docker ps | grep -q postgres; then
    echo "Starting database infrastructure..."
    ./scripts/run-local.sh
    echo "Waiting for services to initialize..."
    sleep 30
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r src/requirements.txt

# Set environment variables
export API_BASE_URL=http://localhost:8000

echo ""
echo "Starting services..."
echo ""

# Start API service
echo "1. Starting ForecastWater API (port 8000)..."
cd src/analytics
python api.py &
API_PID=$!
cd ../..

# Wait for API to start
sleep 5

# Test API health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   API service: READY"
else
    echo "   API service: FAILED"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

# Start dashboard
echo "2. Starting Dashboard (port 8501)..."
streamlit run src/dashboard/app.py &
DASHBOARD_PID=$!

echo ""
echo "Demo Ready!"
echo "==========="
echo ""
echo "Access Points:"
echo "  Dashboard:        http://localhost:8501"
echo "  API Docs:         http://localhost:8000/docs"
echo "  API Health:       http://localhost:8000/health"
echo ""
echo "Features:"
echo "  - Real-time watering predictions"
echo "  - Interactive API documentation"
echo "  - Sensor status monitoring"
echo "  - Historical data visualization"
echo ""
echo "Process IDs: API=$API_PID Dashboard=$DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running
trap "echo 'Stopping services...'; kill $API_PID $DASHBOARD_PID 2>/dev/null || true; exit 0" INT
wait
