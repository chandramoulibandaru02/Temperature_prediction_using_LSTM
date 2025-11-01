# Use an official Python slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/WeatherPrediction.keras

WORKDIR /app

# System deps required for some Python packages and TensorFlow runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libnss3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps (no-cache to keep image small)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# create a folder for the model to be mounted (optional)
RUN mkdir -p /app/model

# Expose port
EXPOSE 5000

# Run the app (dev friendly). If you prefer production, switch to gunicorn.
CMD ["python", "app.py"]
