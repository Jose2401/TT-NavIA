"""
detector.py
-----------
Usa YOLOv8-seg (modelo de segmentación) para obtener:
  - Bounding box
  - Máscara de segmentación (silueta real del objeto)
  - Clase y confianza

La CNN interna de YOLO tiene dos cabezas:
  1. Detección   → bounding boxes
  2. Segmentación → máscaras de 160×160 upscaleadas al frame
"""

from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    # yolov8n-seg.pt  → nano  (rápido, menos preciso)
    # yolov8s-seg.pt  → small (buen balance para TT)
    def __init__(self, model_path: str = "yolov8s-seg.pt", confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Retorna lista de detecciones con:
          label, confidence, bbox (x1,y1,x2,y2), mask (H×W bool array | None)
        """
        results = self.model(frame, stream=False, verbose=False)
        detections = []

        for r in results:
            masks = r.masks  # puede ser None si el modelo no segmenta
            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls   = int(box.cls[0])
                label = self.model.names[cls]

                # Máscara de segmentación (binaria, mismo tamaño que el frame)
                mask = None
                if masks is not None and i < len(masks.data):
                    # masks.data[i] es tensor float [H, W] → binarizar
                    mask_tensor = masks.data[i].cpu().numpy()
                    mask = (mask_tensor > 0.5).astype(np.uint8)  # 0 ó 1

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "mask":       mask,          # np.ndarray H×W uint8 | None
                })

        return detections