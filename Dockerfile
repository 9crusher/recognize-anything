# Recognize Anything Plus Model (RAM++) - Simplified Docker Setup
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml ./
COPY ram/ ./ram/
RUN pip install --no-cache-dir -e .

# Download model checkpoint
RUN mkdir -p /models && \
    wget -q --show-progress -O /models/ram_plus_swin_large_14m.pth \
    https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# Copy inference script
COPY inference_ram_plus_openset.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default working directory for user data
WORKDIR /data

# Default command shows usage
CMD ["python", "/app/inference_ram_plus_openset.py", "--help"]
