"""
motion.py

--------- para no usar el tracker de liberia
Dos niveles de detección de movimiento:

  1. GlobalMotionDetector  → diferencia de frames completos (fondo vs escena)
  2. ObjectMotionTracker   → rastrea centroide de cada objeto entre frames
                             para saber SI ESE OBJETO específico se mueve

El tracker usa IoU para asociar detecciones consecutivas sin depender
de un tracking externo (no requiere ByteTrack/DeepSORT).
"""

import cv2
import numpy as np
from collections import deque


#Movimiento global de escena 

class GlobalMotionDetector:
    def __init__(self, threshold: int = 25, ratio_threshold: float = 0.02):
        self.prev_frame       = None
        self.threshold        = threshold
        self.ratio_threshold  = ratio_threshold

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Retorna (is_moving: bool, motion_ratio: float 0–1)
        motion_ratio normalizado → independiente de resolución.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        diff   = cv2.absdiff(self.prev_frame, gray)
        _, thr = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        total_pixels = gray.shape[0] * gray.shape[1]
        ratio        = float(thr.sum()) / (255 * total_pixels)

        self.prev_frame = gray
        return ratio > self.ratio_threshold, round(ratio, 4)


# Movimiento por objeto (centroide tracking)

def _centroid(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def _iou(b1: tuple, b2: tuple) -> float:
    """Intersection over Union de dos bboxes (x1,y1,x2,y2)."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1    = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2    = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ObjectMotionTracker:
    """
    Mantiene un historial de centroides por objeto.
    Asocia detecciones frame a frame por IoU de bboxes.
    """

    def __init__(self, history_len: int = 6, move_threshold: float = 8.0):
        self.history: dict[int, deque] = {} 
        self.prev_bboxes: list[tuple] = []
        self.next_id   = 0
        self.history_len    = history_len
        self.move_threshold = move_threshold  # px de desplazamiento para "en movimiento"

    def update(self, detections: list[dict]) -> list[bool]:
        """
        Recibe lista de detecciones del frame actual.
        Retorna lista de bools: True si ese objeto está en movimiento.
        """
        current_bboxes = [d["bbox"] for d in detections]
        is_moving      = [False] * len(detections)

        if not self.prev_bboxes:
            for bbox in current_bboxes:
                self._register(bbox)
            self.prev_bboxes = current_bboxes
            return is_moving
    
        matched_prev = set()
        matched_curr = set()
        pairs = []

        for ci, cb in enumerate(current_bboxes):
            best_iou, best_pi = 0.0, -1
            for pi, pb in enumerate(self.prev_bboxes):
                if pi in matched_prev:
                    continue
                score = _iou(cb, pb)
                if score > best_iou:
                    best_iou, best_pi = score, pi
            if best_iou > 0.2:   # umbral de asociación
                pairs.append((ci, best_pi))
                matched_curr.add(ci)
                matched_prev.add(best_pi)

        new_history: dict[int, deque] = {}
        prev_ids    = list(self.history.keys())

        for ci, pi in pairs:
            obj_id = prev_ids[pi] if pi < len(prev_ids) else self.next_id
            dq     = self.history.get(obj_id, deque(maxlen=self.history_len))
            dq.append(_centroid(current_bboxes[ci]))
            new_history[obj_id] = dq

            if len(dq) >= 2:
                dx = dq[-1][0] - dq[0][0]
                dy = dq[-1][1] - dq[0][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist > self.move_threshold:
                    is_moving[ci] = True

        for ci in range(len(current_bboxes)):
            if ci not in matched_curr:
                obj_id = self.next_id
                self.next_id += 1
                dq = deque(maxlen=self.history_len)
                dq.append(_centroid(current_bboxes[ci]))
                new_history[obj_id] = dq

        self.history     = new_history
        self.prev_bboxes = current_bboxes
        return is_moving

    def _register(self, bbox: tuple):
        dq = deque(maxlen=self.history_len)
        dq.append(_centroid(bbox))
        self.history[self.next_id] = dq
        self.next_id += 1