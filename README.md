# Car Damage Detection System

## Overview
Real-time car damage detection system using YOLO (You Only Look Once) deep learning model with Streamlit web interface. The system processes video streams to identify and classify various types of car damage in real-time.

## Features
- Real-time video processing with YOLO object detection
- Interactive web interface using Streamlit
- Dynamic damage label tracking (current frame + cumulative)
- Configurable confidence threshold
- Visual bounding box annotations
- Responsive UI with color-coded damage categories

## System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   YOLO Model     │───▶│   Streamlit UI  │
│   (MP4/Camera)  │    │   Processing     │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Damage Labels  │
                       │   & Bounding Box │
                       └──────────────────┘
```

## Technical Stack
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: Ultralytics YOLO
- **Language**: Python 3.8+

## Quick Start
```bash
pip install -r requirements.txt
streamlit run stcar.py
```

## Project Structure
```
cardamage/
├── stcar.py                 # Main application
├── requirements.txt         # Dependencies
├── docs/                   # Documentation
├── models/                 # YOLO model files
├── datasets/              # Training datasets
├── extracted_frames/      # Video frame extraction
└── runs/                  # Training results
```