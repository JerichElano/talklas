# Build stage
FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10/site-packages -name "__pycache__" -delete

# Final stage
FROM python:3.10-slim

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsndfile.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/bin/ffmpeg /usr/bin/
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/

WORKDIR /app

# Create HF cache directory and install minimal requirements
RUN mkdir -p /tmp/hf_cache \
    && chmod -R 777 /tmp/hf_cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HOME=/root \
    HF_HOME=/tmp/hf_cache \
    PYTHONPATH=/usr/local/lib/python3.10/site-packages \
    PATH="/usr/local/bin:/usr/bin:$PATH"

# Copy your app files
COPY app.py model_manager.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]