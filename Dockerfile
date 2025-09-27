# Dockerfile for Streamlit Dashboard on Cloud Run
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set environment variables for cloud mode
ENV DASHBOARD_MODE=cloud
ENV PORT=8080

# Expose port
EXPOSE 8080

# Note: Health check removed - Cloud Run has built-in health checks

# Run Streamlit
CMD streamlit run src/dashboard/app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
