import os

from ultralytics import YOLO
import numpy as np


_ncnn_threads = None


def _limit_ncnn_threads(n: int) -> None:
    """ultralytics crea la red ncnn (ncnn.Net()) hasta la primera
    inferencia y con los hilos por defecto (4). el numero de hilos se
    tiene que fijar ANTES de cargar los pesos, asi que se envuelve
    ncnn.Net pa ponerlo al crearla."""
    global _ncnn_threads
    if _ncnn_threads is not None or n <= 0:
        return
    try:
        import ncnn
    except ImportError:
        return
    _ncnn_threads = n
    original = ncnn.Net

    def Net(*args, **kwargs):
        net = original(*args, **kwargs)
        net.opt.num_threads = _ncnn_threads
        return net
    ncnn.Net = Net


def _exported_imgsz(model_path: str):
    """(alto, ancho) de un modelo exportado por ultralytics; None si es .pt"""
    meta = os.path.join(model_path, "metadata.yaml")
    if not os.path.isdir(model_path) or not os.path.exists(meta):
        return None
    try:
        import yaml
        with open(meta, encoding="utf-8") as fh:
            imgsz = yaml.safe_load(fh).get("imgsz")
        return tuple(imgsz) if imgsz else None
    except Exception:
        return None


class ObjectDetector:
    # yolov8n-seg.pt  = nano  (rápido, menos preciso, probable modelo para la pi xd)
    # yolov8s-seg.pt  = small (mas pro ojala funcione en la pi)
    def __init__(self, model_path: str = "yolov8s-seg.pt", confidence: float = 0.4,
                 task: str = None, masks: bool = True):
        # model_path tambien puede ser un modelo exportado (carpeta
        # *_ncnn_model, .onnx): en la pi el ncnn corre ~4x mas rapido que
        # el .pt con los mismos pesos. esos traen el tamaño de entrada fijo
        # y hay que decirles la tarea
        self.imgsz = _exported_imgsz(model_path)
        if task is None and self.imgsz is not None:
            task = "segment" if "seg" in os.path.basename(
                os.path.normpath(model_path)) else "detect"
        self.model = YOLO(model_path, task=task) if task else YOLO(model_path)
        self.confidence = confidence
        self.masks = masks   # False: no sacar las mascaras (nadie las usa en la pi)
        # hilos del ncnn: en la pi 5 con 3 va igual de rapido que con 4 y
        # deja un nucleo pa la voz, el microfono y los sensores (con 4 y
        # la voz sintetizando, la vision se iba de 380 a 1160 ms)
        if self.imgsz is not None:
            _limit_ncnn_threads(int(os.environ.get("NAVIA_HILOS_VISION", "3")))

    def predict_kwargs(self) -> dict:
        return {"imgsz": self.imgsz} if self.imgsz is not None else {}

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(frame, stream=False, verbose=False,
                             **self.predict_kwargs())
        detections = []

        for r in results:
            masks = r.masks if self.masks else None  # puede ser None si el modelo no segmenta
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