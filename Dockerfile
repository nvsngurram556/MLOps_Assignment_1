# Multi-stage Dockerfile for California Housing ML API

# Stage 1: Build stage
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r mlapi && useradd -r -g mlapi mlapi

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/mlapi/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY artifacts/ ./artifacts/

# Set environment variables
ENV PATH=/home/mlapi/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/artifacts

# Change ownership to mlapi user
RUN chown -R mlapi:mlapi /app

# Switch to non-root user
USER mlapi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]