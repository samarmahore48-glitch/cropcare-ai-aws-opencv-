# ==============================================================================
# CropCare AI - AWS EC2 Graviton (ARM64) & Multi-Arch Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output for CloudWatch logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    AWS_DEFAULT_REGION=us-east-1

WORKDIR /app

# Install system dependencies for OpenCV and high-performance SIMD
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/cropcare_ai/backend/
WORKDIR /app/cropcare_ai/backend

# Create non-root system user for security best practices
RUN useradd -m -u 1001 cropcare && chown -R cropcare:cropcare /app
USER cropcare

EXPOSE 8000

# Healthcheck for AWS Application Load Balancer (ALB)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch production server with Uvicorn worker pool
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
