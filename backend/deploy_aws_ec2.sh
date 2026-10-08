#!/bin/bash
# ==============================================================================
# CropCare AI - AWS EC2 Graviton Deployment Script (AWS Sponsored Track)
# Target: Amazon Linux 2023 or Ubuntu on AWS Graviton3 (c7g.xlarge / t4g.xlarge)
# ==============================================================================

set -e

echo "=== [1/6] Detecting System Architecture ==="
ARCH=$(uname -m)
echo "Architecture detected: $ARCH"
if [ "$ARCH" = "aarch64" ]; then
    echo ">> Running on native AWS Graviton (ARM64 Neoverse). NEON SIMD enabled for OpenCV 5."
else
    echo ">> Running on x86_64 architecture."
fi

echo "=== [2/6] Updating System & Installing Prerequisites ==="
if [ -f /etc/amazon-linux-release ]; then
    sudo dnf update -y
    sudo dnf install -y python3.11 python3.11-pip git libglvnd-glx
elif [ -f /etc/lsb-release ]; then
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 curl
fi

echo "=== [3/6] Setting Up Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [4/6] Configuring AWS IAM & Cloud Resources ==="
# S3 Bucket Setup (Idempotent)
S3_BUCKET=${CROPCARE_S3_BUCKET:-"cropcare-farmer-leaf-uploads-prod"}
aws s3 mb s3://$S3_BUCKET --region us-east-1 2>/dev/null || true

# DynamoDB Table Setup (Idempotent)
aws dynamodb create-table \
    --table-name CropCare_FarmScans \
    --attribute-definitions AttributeName=farm_id,AttributeType=S AttributeName=scan_id,AttributeType=S \
    --key-schema AttributeName=farm_id,KeyType=HASH AttributeName=scan_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1 2>/dev/null || true

echo "=== [5/6] Creating Systemd Service for Auto-Restart ==="
cat << 'EOF' | sudo tee /etc/systemd/system/cropcare.service
[Unit]
Description=CropCare AI Vision & Cloud Pipeline
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/cropcare_ai/backend
ExecStart=/opt/cropcare_ai/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
Environment=AWS_DEFAULT_REGION=us-east-1
Environment=CROPCARE_S3_BUCKET=cropcare-farmer-leaf-uploads-prod

[Install]
WantedBy=multi-user.target
EOF

echo "=== [6/6] Launching CropCare AI Service ==="
sudo systemctl daemon-reload
sudo systemctl enable cropcare
sudo systemctl restart cropcare

echo ">> Deployment complete! Liveness check:"
curl -s http://localhost:8000/api/v1/health | grep "healthy" && echo ">> Status: ONLINE"
