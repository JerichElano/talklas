FROM python:3.10-slim

# Install system dependencies for torchaudio and soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the Hugging Face cache directory and set permissions
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

# Set environment variables for Hugging Face cache
ENV HOME=/root
ENV HF_HOME=/tmp/hf_cache

# Copy the rest of your app's code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
