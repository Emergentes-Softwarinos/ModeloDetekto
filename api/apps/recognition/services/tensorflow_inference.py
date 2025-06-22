import os
import cv2
from ultralytics import YOLO
import numpy as np


class YOLOInference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.labels = self.model.names

    def run_inference(self, frame):
        results = self.model(frame, verbose=False)
        detections = results[0].boxes
        objects_detected = []

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = self.labels[classidx]
            conf = detections[i].conf.item()

            if conf > 0.5:
                objects_detected.append(
                    {
                        "class": classname,
                        "confidence": conf,
                        "bbox": [xmin, ymin, xmax, ymax],
                    }
                )

        return objects_detected
