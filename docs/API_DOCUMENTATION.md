# API Documentation - Car Damage Detection System

## Core API Reference

### CarDamageDetector Class

#### Constructor
```python
CarDamageDetector(config_path="config/config.yaml")
```
**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:** CarDamageDetector instance

#### Methods

### `process_frame(frame)`
Process single video frame for damage detection.

**Parameters:**
- `frame` (numpy.ndarray): Input frame in BGR format (H, W, 3)

**Returns:**
- `processed_frame` (numpy.ndarray): Frame with bounding boxes and labels
- `detections` (list): List of detection dictionaries

**Example:**
```python
detector = CarDamageDetector()
frame = cv2.imread('car_image.jpg')
processed_frame, detections = detector.process_frame(frame)
```

### `process_video(video_path, output_path=None)`
Process entire video file for damage detection.

**Parameters:**
- `video_path` (str): Path to input video file
- `output_path` (str, optional): Path to save processed video

**Returns:**
- `results` (dict): Processing results with statistics

**Example:**
```python
results = detector.process_video('input.mp4', 'output.mp4')
print(f"Total detections: {results['total_detections']}")
```

### `get_model_info()`
Get information about the loaded YOLO model.

**Returns:**
- `model_info` (dict): Model metadata and statistics

**Example:**
```python
info = detector.get_model_info()
print(f"Model classes: {info['classes']}")
print(f"Model size: {info['parameters']} parameters")
```

## Detection Object Structure

### Detection Dictionary
```python
{
    'bbox': [x1, y1, x2, y2],      # Bounding box coordinates
    'confidence': 0.85,             # Detection confidence (0.0-1.0)
    'class': 2,                     # Class ID
    'label': 'scratch',             # Class name
    'area': 1250.5,                # Bounding box area
    'center': [320, 240]            # Bounding box center
}
```

### Results Dictionary
```python
{
    'total_detections': 45,         # Total number of detections
    'unique_classes': ['scratch', 'dent'],  # Unique damage types found
    'processing_time': 12.5,        # Total processing time (seconds)
    'fps': 30.0,                    # Processing frame rate
    'frame_count': 375,             # Total frames processed
    'confidence_stats': {           # Confidence statistics
        'mean': 0.72,
        'std': 0.15,
        'min': 0.51,
        'max': 0.98
    }
}
```

## Configuration API

### Configuration File Structure
```yaml
model:
  path: "models/allyolov8best.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  input_size: 640
  device: "auto"  # "cpu", "cuda", or "auto"

processing:
  batch_size: 1
  max_detections: 100
  enable_tracking: false

video:
  input_formats: [".mp4", ".avi", ".mov"]
  output_format: "mp4"
  fps: 30
  quality: "high"

ui:
  title: "Car Damage Detection"
  theme: "dark"
  show_confidence: true
  show_fps: true
  color_scheme:
    scratch: "#FF5722"
    dent: "#2196F3"
    crack: "#FF9800"
    rust: "#795548"
    broken_part: "#F44336"
```

### Configuration Methods

#### `load_config(config_path)`
Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `config` (dict): Configuration dictionary

#### `update_config(updates)`
Update configuration parameters.

**Parameters:**
- `updates` (dict): Configuration updates

**Example:**
```python
detector.update_config({
    'model': {'confidence_threshold': 0.7},
    'ui': {'theme': 'light'}
})
```

## Utility Functions

### `format_labels(label_set, color="#4CAF50", newline=False)`
Format detection labels for display.

**Parameters:**
- `label_set` (set): Set of damage labels
- `color` (str): Background color for labels
- `newline` (bool): Add line breaks between labels

**Returns:**
- `formatted_html` (str): HTML formatted label string

### `calculate_damage_severity(detections)`
Calculate overall damage severity score.

**Parameters:**
- `detections` (list): List of detection dictionaries

**Returns:**
- `severity_score` (float): Severity score (0.0-1.0)
- `severity_level` (str): "Low", "Medium", "High", or "Critical"

**Example:**
```python
score, level = calculate_damage_severity(detections)
print(f"Damage severity: {level} ({score:.2f})")
```

### `export_results(detections, format="json", output_path=None)`
Export detection results to various formats.

**Parameters:**
- `detections` (list): Detection results
- `format` (str): Export format ("json", "csv", "xml")
- `output_path` (str, optional): Output file path

**Returns:**
- `export_path` (str): Path to exported file

## Error Handling

### Exception Classes

#### `ModelLoadError`
Raised when YOLO model fails to load.

```python
try:
    detector = CarDamageDetector()
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
```

#### `VideoProcessingError`
Raised when video processing encounters errors.

```python
try:
    results = detector.process_video('video.mp4')
except VideoProcessingError as e:
    print(f"Video processing failed: {e}")
```

#### `ConfigurationError`
Raised when configuration is invalid.

```python
try:
    detector.load_config('invalid_config.yaml')
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Performance Monitoring

### `get_performance_stats()`
Get current performance statistics.

**Returns:**
- `stats` (dict): Performance metrics

```python
{
    'memory_usage': 2048.5,        # Memory usage in MB
    'cpu_usage': 25.3,             # CPU usage percentage
    'gpu_usage': 45.7,             # GPU usage percentage (if available)
    'inference_time': 15.2,        # Average inference time (ms)
    'fps': 28.5,                   # Current processing FPS
    'total_frames': 1250           # Total frames processed
}
```

### `reset_performance_counters()`
Reset performance monitoring counters.

**Example:**
```python
detector.reset_performance_counters()
stats = detector.get_performance_stats()
```

## Batch Processing API

### `process_batch(frames, batch_size=4)`
Process multiple frames in batches for improved performance.

**Parameters:**
- `frames` (list): List of input frames
- `batch_size` (int): Number of frames per batch

**Returns:**
- `batch_results` (list): List of detection results for each frame

### `process_image_directory(directory_path, output_path=None)`
Process all images in a directory.

**Parameters:**
- `directory_path` (str): Path to directory containing images
- `output_path` (str, optional): Output directory for processed images

**Returns:**
- `results` (dict): Processing results summary

## Streaming API

### `start_stream(source="webcam")`
Start real-time video stream processing.

**Parameters:**
- `source` (str): Stream source ("webcam", "rtsp://url", or device index)

**Returns:**
- `stream_id` (str): Unique stream identifier

### `stop_stream(stream_id)`
Stop active video stream.

**Parameters:**
- `stream_id` (str): Stream identifier to stop

### `get_stream_frame(stream_id)`
Get latest processed frame from stream.

**Parameters:**
- `stream_id` (str): Stream identifier

**Returns:**
- `frame` (numpy.ndarray): Latest processed frame
- `detections` (list): Latest detections

## Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)
detector = CarDamageDetector()

@app.route('/detect', methods=['POST'])
def detect_damage():
    # Decode base64 image
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame
    _, detections = detector.process_frame(frame)
    
    return jsonify({
        'detections': detections,
        'count': len(detections)
    })
```

### WebSocket Integration
```python
import asyncio
import websockets
import json

async def handle_client(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        
        if data['type'] == 'process_frame':
            # Process frame and send results
            results = detector.process_frame(data['frame'])
            await websocket.send(json.dumps({
                'type': 'detection_results',
                'data': results
            }))
```