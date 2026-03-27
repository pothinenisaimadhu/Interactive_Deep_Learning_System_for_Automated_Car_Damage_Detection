# Troubleshooting Guide - Car Damage Detection System

## Common Issues and Solutions

### 1. NumPy Version Compatibility Error

**Error Message:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5 as it may crash.
ImportError: numpy.core.multiarray failed to import
```

**Solution Options:**

#### Option A: Downgrade NumPy (Recommended)
```bash
pip install "numpy<2.0"
```

#### Option B: Create Fresh Environment
```bash
# Create new virtual environment
python -m venv car_damage_env
car_damage_env\Scripts\activate

# Install compatible versions
pip install numpy==1.24.3
pip install -r requirements.txt
```

#### Option C: Use Conda Environment
```bash
conda create -n car_damage python=3.9
conda activate car_damage
conda install numpy=1.24.3
pip install -r requirements.txt
```

### 2. YOLO Model Loading Issues

**Error:** Model file not found or corrupted

**Solutions:**
```python
# Check if model file exists
import os
model_path = r"C:\Users\saima\Downloads\allyolov8best.pt"
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    # Download default model
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # Downloads automatically
```

### 3. Video File Loading Problems

**Error:** Video file cannot be opened

**Solutions:**
```python
import cv2

def check_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f}s")
    cap.release()
    return True

# Usage
video_path = r"C:\Users\saima\Downloads\WhatsApp Video 2025-04-28 at 10.52.16_7c397688.mp4"
check_video_file(video_path)
```

### 4. Streamlit Port Issues

**Error:** Port 8501 already in use

**Solutions:**
```bash
# Use different port
streamlit run stcar.py --server.port 8502

# Kill existing Streamlit processes (Windows)
taskkill /f /im streamlit.exe

# Kill existing processes (Linux/Mac)
pkill -f streamlit
```

### 5. Memory Issues

**Error:** Out of memory during processing

**Solutions:**
```python
# Reduce video resolution
def resize_frame(frame, scale=0.5):
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

# Process every nth frame
frame_skip = 2  # Process every 2nd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    # Process frame
    processed_frame, detections = process_frame(frame)
```

### 6. GPU/CUDA Issues

**Error:** CUDA out of memory or not available

**Solutions:**
```python
import torch

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

# Force CPU usage if GPU issues persist
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### 7. Import Errors

**Error:** Module not found

**Solutions:**
```bash
# Install missing packages
pip install opencv-python
pip install ultralytics
pip install streamlit

# Upgrade pip
python -m pip install --upgrade pip

# Install from requirements
pip install -r requirements.txt --upgrade
```

### 8. Performance Issues

**Symptoms:** Slow processing, high CPU usage

**Solutions:**
```python
# Optimize processing
def optimize_processing():
    # Reduce confidence threshold for faster processing
    CONFIDENCE_THRESHOLD = 0.3
    
    # Use smaller model
    model = YOLO('yolov8n.pt')  # Nano model (fastest)
    
    # Reduce input size
    def preprocess_frame(frame):
        return cv2.resize(frame, (320, 320))  # Smaller input
    
    # Skip frames
    frame_skip = 3
    
    return CONFIDENCE_THRESHOLD, model, frame_skip
```

## Environment Setup Verification

### Check Python Environment
```python
import sys
import cv2
import streamlit
import ultralytics
import numpy as np
import torch

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("Streamlit version:", streamlit.__version__)
print("Ultralytics version:", ultralytics.__version__)
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
```

### System Requirements Check
```python
import psutil
import platform

def check_system_requirements():
    # System info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # CPU
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"CPU usage: {psutil.cpu_percent()}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Disk space: {disk.free / (1024**3):.1f} GB free")

check_system_requirements()
```

## Quick Fix Script

Create this script to automatically fix common issues:

```python
# fix_environment.py
import subprocess
import sys
import os

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command: {command}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False

def fix_numpy_issue():
    print("Fixing NumPy compatibility issue...")
    commands = [
        "pip uninstall numpy -y",
        "pip install numpy==1.24.3",
        "pip install --upgrade streamlit",
        "pip install --upgrade ultralytics"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"Failed to execute: {cmd}")
            return False
    
    print("NumPy issue fixed!")
    return True

def install_requirements():
    print("Installing requirements...")
    return run_command("pip install -r requirements.txt")

def main():
    print("Car Damage Detection - Environment Fix Script")
    print("=" * 50)
    
    # Fix NumPy issue
    if not fix_numpy_issue():
        print("Failed to fix NumPy issue")
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        return
    
    print("Environment setup completed successfully!")
    print("You can now run: streamlit run stcar.py")

if __name__ == "__main__":
    main()
```

## Running the Application

### Step-by-Step Startup
```bash
# 1. Activate environment
car_damage_env\Scripts\activate

# 2. Verify installation
python -c "import streamlit, cv2, ultralytics; print('All modules imported successfully')"

# 3. Check model file
python -c "import os; print('Model exists:', os.path.exists(r'C:\Users\saima\Downloads\allyolov8best.pt'))"

# 4. Run application
streamlit run stcar.py
```

### Alternative Startup Methods
```bash
# Method 1: Direct Python execution
python -m streamlit run stcar.py

# Method 2: With specific configuration
streamlit run stcar.py --server.port 8501 --server.address localhost

# Method 3: Debug mode
streamlit run stcar.py --logger.level debug
```