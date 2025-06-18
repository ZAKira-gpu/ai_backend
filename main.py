from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")  # Or your custom model path

@app.get("/")
def read_root():
    return {"status": "YOLOv8 FastAPI is running 🚀"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        results = model.predict(np.array(image))

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    },
                }
                detections.append(detection)

        return {"detections": detections}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
