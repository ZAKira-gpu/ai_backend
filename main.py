from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = YOLO("yolov8n.pt")  # Upload this to Render

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model(img)
    detections = results[0].boxes.xyxy.tolist()  # x1, y1, x2, y2
    confidences = results[0].boxes.conf.tolist()
    class_ids = results[0].boxes.cls.tolist()
    return JSONResponse({
        "boxes": detections,
        "confidences": confidences,
        "class_ids": class_ids
    })
