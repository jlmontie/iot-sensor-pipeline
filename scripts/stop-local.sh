#!/bin/bash
# Stop local IoT pipeline processes and services

echo "Stopping local IoT pipeline..."

# Stop background Python processes
echo "Stopping data generator and frontload processes..."
pkill -f "simulate_stream.py" 2>/dev/null && echo "✓ Data generator stopped"
pkill -f "frontload-dashboard-data.py" 2>/dev/null && echo "✓ Frontload process stopped"

# Stop Docker services
echo "Stopping Docker services..."
docker compose down

echo "Local pipeline stopped."
