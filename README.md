# Football Video Processing for YOLO

## Overview
This project processes football video data to make it compatible with the YOLO object detection format. It involves extracting frames from videos, generating corresponding annotations, and preparing the dataset for YOLO-based object detection tasks.

## Features
- Extract frames from football videos.
- Generate YOLO-compatible annotations for each frame.
- Prepare dataset for YOLO object detection training and inference.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV
- YOLO (e.g., YOLOv5, YOLOv8)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Extract Frames from Video
Run the following script to extract frames from a video:
```bash
python prepare_data.py/
```

### 2. Train YOLO Model
Use the prepared dataset to train a YOLO model:
```bash
python train.py --data dataset.yaml --weights yolov5s.pt --epochs 50
```

## Folder Structure
```
FootballAnalytics/
│── datasets/
│   ├── football_train/       # Raw football videos
│   ├── football_test/       
│   ├── football_dataset/      # After done processing
│      |── images/
│        |── train/
│        |── test/
│      |── labels/
│        |── train/
│        |── test/
│── prepare_data.py  # Frame extraction script
│── generate_annotations.py  # Annotation conversion script
│── models/
│── README.md
```

