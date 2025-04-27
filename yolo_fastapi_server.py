from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import cv2
import numpy as np

app = FastAPI()
model = YOLO('yolov8n.pt')  # Load YOLOv8 model (you can switch to yolov8s.pt etc.)

@app.get("/")
async def root():
    return {"message": "Welcome to YOLOv8 Detection API!"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img, imgsz=640, conf=0.5)
    detection_data = results[0].tojson()

    return {"detections": detection_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)