FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV PIP_NO_CACHE_DIR=1 \
    TORCH_CPU_ONLY=1 \
    PYTORCH_CUDA_ALLOW_CONTAINER=0
# Install system packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /tmp/matplotlib /tmp/ultralytics && \
    chmod -R 755 /tmp/matplotlib /tmp/ultralytics

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY . .
RUN pip install -r requirements.txt


# If you're using FastAPI with Uvicorn:
EXPOSE 8000
CMD ["uvicorn", "image-tagging:app", "--host", "0.0.0.0", "--port", "8000"]
