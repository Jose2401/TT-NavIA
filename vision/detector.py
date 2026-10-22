from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    # yolov8n-seg.pt  = nano  (rápido, menos preciso, probable modelo para la pi xd)
    # yolov8s-seg.pt  = small (mas pro ojala funcione en la pi)
    def __init__(self, model_path: str = "yolov8s-seg.pt", confidence: float = 0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict]:
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
                    # masks.data[i] es tensor float [H, W] 
                    mask_tensor = masks.data[i].cpu().numpy()
                    mask = (mask_tensor > 0.5).astype(np.uint8)  # 0 ó 1

                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "mask":       mask,       
                })

        return detections