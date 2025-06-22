from fastapi import APIRouter, UploadFile, File
from .services.tensorflow_inference import YOLOInference
import cv2
import numpy as np
from io import BytesIO

router = APIRouter()


@router.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Leer el archivo de imagen
    image_data = await file.read()
    img_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Realizar la inferencia usando YOLO
    model_path = "path_to_your_model.pt"  # Define tu ruta al modelo YOLO
    yolo = YOLOInference(model_path)
    objects = yolo.run_inference(frame)

    return {"objects_detected": objects}
