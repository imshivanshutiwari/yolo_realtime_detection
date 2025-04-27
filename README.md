# Real-Time Object Detection using YOLOv8 (Webcam & Video File)

## Project Overview
This project demonstrates a real-time object detection system using YOLOv8 (You Only Look Once, version 8) with the following powerful features:
- Model selection (YOLOv8n, YOLOv8s, YOLOv8m)
- Confidence threshold slider for live control
- Save detection snapshots with 's' key
- Real-time object counting per class
- Support for both Webcam and Video File input
- Saves output videos and screenshots automatically

## Features
- **Real-Time Detection:** Fast, smooth object detection using YOLOv8.
- **Model Switcher:** Choose between different YOLO models at startup.
- **Confidence Slider:** Adjust minimum detection confidence live.
- **Save Screenshots:** Press 's' to capture and save frames.
- **Object Counter:** Counts and displays detected objects live.
- **Video Support:** Analyze saved videos, not just live camera feed.

## Requirements
- Python 3.9 or higher
- Libraries:
  - ultralytics
  - opencv-python
  - numpy
- Install with:
  ```bash
  pip install -r requirements.txt

  How to Run

1. Clone the repository.


2. Install required libraries.


3. Run:

python yolo_realtime.py


4. Follow terminal instructions:

Choose model (YOLOv8n/s/m)

Choose input (Webcam or Video file)




Folder Structure

yolo_realtime_detection/
├── output_videos/     # Saved output videos
├── results/           # Saved screenshots
├── yolo_realtime.py   # Main detection script
├── README.md
├── requirements.txt

Example Screenshots

(Insert some screenshots of detection results here.)

Credits

YOLOv8 by Ultralytics

OpenCV-Python

PyTorch



## Author
Shivanshu Tiwari