import cv2 as cv
import supervision as sv
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import base64

app = FastAPI()

# Load YOLO model once when app starts
MODEL_PATH = "model.pt"
model = YOLO(MODEL_PATH)
class_dict = model.names

# Accept base64-encoded image
class ImageInput(BaseModel):
    image_base64: str


def count_items(input_list: list):
    counts = {}
    for item in input_list:
        key = str(item).lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def image_prediction(image_path: str, confidence: float = 0.5):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")

    result = model(img)[0]
    detections = sv.Detections.from_ultralytics(result)

    tags = []
    if detections.class_id is not None:
        detections = detections[(detections.confidence > confidence)]
        tags = [f"{class_dict[cls_id]}" for cls_id in detections.class_id]

    return tags


@app.get("/")
async def root():
    return {"message": "Wecome to image prediction"}


@app.post("/predict")
async def predict(input_data: ImageInput):
    try:
        # Decode base64 image to binary
        image_data = base64.b64decode(input_data.image_base64)

        # Save binary image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        # Perform prediction
        tags = image_prediction(tmp_path)
        os.remove(tmp_path)

        return JSONResponse(content={"tags": tags, "count": count_items(tags)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
