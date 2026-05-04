# ---------------------------------------------------------------------------
# Dockerfile for Predictive Cluster Manager API
# ---------------------------------------------------------------------------
# This runs the FastAPI backend inside a container.
# The container needs access to the host Docker daemon to manage
# simulated cluster nodes (via a socket mount in docker-compose).
# ---------------------------------------------------------------------------

FROM python:3.11-slim

LABEL maintainer="Predictive Cluster Manager"
LABEL description="AI-powered cloud cluster autoscaler API"

WORKDIR /app

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artefacts.
COPY astar_scaler.py .
COPY docker_manager.py .
COPY api.py .
COPY workload_lstm_model.keras .
COPY input_scaler.pkl .
COPY target_scaler.pkl .
COPY prediction_calibration.pkl .
COPY prediction_postprocess_config.pkl .
COPY spike_guard_config.pkl .

# Expose the API port.
EXPOSE 8000

# Health check.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
