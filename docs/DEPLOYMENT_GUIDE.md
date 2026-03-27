# Deployment Guide - Car Damage Detection System

## Local Development Setup

### Prerequisites
- Python 3.8-3.11
- Git
- 8GB+ RAM
- NVIDIA GPU (optional, for acceleration)

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd cardamage

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run stcar.py
```

## Production Deployment

### 1. Docker Deployment

#### Basic Docker Setup
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data/videos results

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "stcar.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

#### Build and Run
```bash
# Build image
docker build -t car-damage-detector:latest .

# Run container
docker run -d \
    --name car-damage-app \
    -p 8501:8501 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    car-damage-detector:latest
```

#### GPU Support
```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Continue with previous Dockerfile content...
```

```bash
# Run with GPU support
docker run -d \
    --gpus all \
    --name car-damage-gpu \
    -p 8501:8501 \
    car-damage-detector:gpu
```

### 2. Docker Compose Deployment

#### docker-compose.yml
```yaml
version: '3.8'

services:
  car-damage-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./results:/app/results
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/models/allyolov8best.pt
      - CONFIDENCE_THRESHOLD=0.5
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - car-damage-app
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server car-damage-app:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://streamlit;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 3. Kubernetes Deployment

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: car-damage-detector
  labels:
    app: car-damage-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: car-damage-detector
  template:
    metadata:
      labels:
        app: car-damage-detector
    spec:
      containers:
      - name: car-damage-app
        image: car-damage-detector:latest
        ports:
        - containerPort: 8501
        env:
        - name: MODEL_PATH
          value: "/app/models/allyolov8best.pt"
        - name: CONFIDENCE_THRESHOLD
          value: "0.5"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: car-damage-service
spec:
  selector:
    app: car-damage-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
```

#### Deploy to Kubernetes
```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -l app=car-damage-detector
kubectl get services

# Scale deployment
kubectl scale deployment car-damage-detector --replicas=5
```

## Cloud Platform Deployments

### 1. AWS Deployment

#### ECS Fargate
```json
{
  "family": "car-damage-detector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "car-damage-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/car-damage-detector:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/allyolov8best.pt"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/car-damage-detector",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EC2 with Auto Scaling
```bash
#!/bin/bash
# user-data.sh for EC2 instances

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone application
git clone <repository-url> /opt/car-damage-detector
cd /opt/car-damage-detector

# Start application
docker-compose up -d

# Setup log rotation
cat > /etc/logrotate.d/car-damage << EOF
/opt/car-damage-detector/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF
```

### 2. Google Cloud Platform

#### Cloud Run Deployment
```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: car-damage-detector
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: gcr.io/project-id/car-damage-detector:latest
        ports:
        - containerPort: 8501
        env:
        - name: MODEL_PATH
          value: "/app/models/allyolov8best.pt"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

```bash
# Deploy to Cloud Run
gcloud run deploy car-damage-detector \
    --image gcr.io/project-id/car-damage-detector:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

### 3. Azure Container Instances

#### ARM Template
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-03-01",
      "name": "car-damage-detector",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "car-damage-app",
            "properties": {
              "image": "your-registry.azurecr.io/car-damage-detector:latest",
              "ports": [
                {
                  "port": 8501,
                  "protocol": "TCP"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              },
              "environmentVariables": [
                {
                  "name": "MODEL_PATH",
                  "value": "/app/models/allyolov8best.pt"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8501,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ]
}
```

## Monitoring and Logging

### 1. Application Monitoring
```python
# monitoring.py
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
DETECTION_COUNTER = Counter('detections_total', 'Total detections', ['damage_type'])
PROCESSING_TIME = Histogram('frame_processing_seconds', 'Frame processing time')
ACTIVE_STREAMS = Gauge('active_streams', 'Number of active video streams')

class MonitoringMixin:
    def __init__(self):
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    @PROCESSING_TIME.time()
    def process_frame_monitored(self, frame):
        start_time = time.time()
        
        try:
            processed_frame, detections = self.process_frame(frame)
            
            # Update metrics
            for detection in detections:
                DETECTION_COUNTER.labels(damage_type=detection['label']).inc()
            
            processing_time = time.time() - start_time
            self.logger.info(f"Processed frame in {processing_time:.3f}s with {len(detections)} detections")
            
            return processed_frame, detections
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            raise
```

### 2. Health Checks
```python
# health.py
from flask import Flask, jsonify
import psutil
import torch

health_app = Flask(__name__)

@health_app.route('/health')
def health_check():
    try:
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        disk_usage = psutil.disk_usage('/').percent
        
        # Check GPU if available
        gpu_available = torch.cuda.is_available()
        gpu_memory = None
        if gpu_available:
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        
        # Determine health status
        status = "healthy"
        if memory_usage > 90 or cpu_usage > 95 or disk_usage > 95:
            status = "unhealthy"
        
        return jsonify({
            "status": status,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "disk_usage": disk_usage,
            "gpu_available": gpu_available,
            "gpu_memory_usage": gpu_memory
        }), 200 if status == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

if __name__ == '__main__':
    health_app.run(host='0.0.0.0', port=8080)
```

## Security Configuration

### 1. Environment Variables
```bash
# .env
MODEL_PATH=/app/models/allyolov8best.pt
CONFIDENCE_THRESHOLD=0.5
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,your-domain.com
MAX_UPLOAD_SIZE=100MB
RATE_LIMIT=100
```

### 2. Security Headers
```python
# security.py
from functools import wraps
from flask import request, abort
import time

# Rate limiting
request_counts = {}

def rate_limit(max_requests=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            
            # Clean old requests
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip]
                if current_time - req_time < window
            ]
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                abort(429)  # Too Many Requests
            
            request_counts[client_ip].append(current_time)
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator
```

## Backup and Recovery

### 1. Data Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup models
cp -r /app/models $BACKUP_DIR/

# Backup configuration
cp -r /app/config $BACKUP_DIR/

# Backup logs (last 7 days)
find /app/logs -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# Upload to cloud storage (example: AWS S3)
aws s3 cp $BACKUP_DIR.tar.gz s3://your-backup-bucket/

# Clean old backups (keep last 30 days)
find /backups -name "*.tar.gz" -mtime +30 -delete
```

### 2. Disaster Recovery
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
RESTORE_DIR="/app"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
docker-compose down

# Extract backup
tar -xzf $BACKUP_FILE -C /tmp/

# Restore files
cp -r /tmp/backup_*/models $RESTORE_DIR/
cp -r /tmp/backup_*/config $RESTORE_DIR/

# Restart application
docker-compose up -d

echo "Restore completed successfully"
```