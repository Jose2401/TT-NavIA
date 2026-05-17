"""
obstacle_logic.py (PRO EXTENDIDO)
--------------------------------
Mejoras:
✔ Clasificación semántica + heurística
✔ Estabilización de categorías (anti-parpadeo)
✔ Detección de muebles aunque YOLO no los clasifique bien
✔ Alias de etiquetas (mejor compatibilidad COCO)
✔ Estimación de distancia mejorada
"""

# ── Alias para mejorar detecciones YOLO ───────────────────────────────────────
LABEL_ALIASES: dict[str, str] = {
    "tv": "monitor",
    "tvmonitor": "monitor",
    "dining table": "table",
    "couch": "sofa",
    "refrigerator": "fridge",
}

# ── Taxonomía semántica ──────────────────────────────────────────────────────

SIGNAGE_LABELS: set[str] = {
    "stop sign",
    "traffic light",
    "exit sign",
    "emergency sign",
    "no entry sign",
    "fire exit",
}

BARRIER_LABELS: set[str] = {
    "wall",
    "door",
    "building",
    "fence",
    "window",
    "glass",
    "fridge",
}

LARGE_LABELS: set[str] = {
    "table",
    "sofa",
    "bed",
    "car", "truck", "bus",
    "monitor",
    "oven",
    "sink",
}

MEDIUM_LABELS: set[str] = {
    "chair",
    "backpack", "suitcase",
    "person",
    "dog", "cat",
    "bicycle", "motorcycle",
    "potted plant",
    "toilet",
}

MINOR_LABELS: set[str] = {
    "bottle", "cup", "bowl",
    "sports ball",
    "banana", "apple",
    "book",
    "cell phone",
    "remote",
    "shoe",
    "toy",
    "teddy bear",
    "umbrella",
    "handbag",
    "tie",
    "scissors",
    "vase",
    "clock",
    "laptop",
    "mouse",
    "keyboard",
}

CATEGORY_ORDER = [
    "señalización",
    "barrera",
    "obstáculo grande",
    "obstáculo mediano",
    "obstáculo menor"
]

CATEGORY_COLORS = {
    "señalización":     (255, 215, 0),
    "barrera":          (0,   0,   255),
    "obstáculo grande": (0,   165, 255),
    "obstáculo mediano":(0,   255, 255),
    "obstáculo menor":  (0,   255, 0),
    "desconocido":      (180, 180, 180),
}


# ── Clasificador ──────────────────────────────────────────────────────────────

class ObstacleClassifier:

    def __init__(self):
        # memoria para estabilización
        self.prev_classifications = {}

    # ─────────────────────────────────────────────
    def _smooth_ratio(self, ratio: float) -> float:
        return round(ratio, 2)

    # ─────────────────────────────────────────────
    def _is_possible_furniture(self, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        ratio = (w * h) / (frame_w * frame_h)

        # heurística: grande + cerca del suelo
        if ratio > 0.05 and y2 > frame_h * 0.6:
            return True

        return False

    # ─────────────────────────────────────────────
    def classify(self, detection: dict, frame_w: int, frame_h: int) -> str:
        raw_label = detection["label"].lower()
        bbox = detection["bbox"]

        # aplicar alias
        label = LABEL_ALIASES.get(raw_label, raw_label)

        key = tuple(bbox)

        # ── Estabilización (si ya se vio antes) ──
        if key in self.prev_classifications:
            return self.prev_classifications[key]

        # 1. Señalización
        if label in SIGNAGE_LABELS:
            result = "señalización"

        # 2. Barrera
        elif label in BARRIER_LABELS:
            result = "barrera"

        # 3. Clases conocidas
        elif label in LARGE_LABELS:
            result = "obstáculo grande"

        elif label in MEDIUM_LABELS:
            result = "obstáculo mediano"

        elif label in MINOR_LABELS:
            result = "obstáculo menor"

        # 4. Detección heurística de muebles
        elif self._is_possible_furniture(bbox, frame_w, frame_h):
            result = "obstáculo grande"

        # 5. Fallback por tamaño
        else:
            result = self._classify_by_size(bbox, frame_w, frame_h)

        # guardar en memoria
        self.prev_classifications[key] = result

        return result

    # ─────────────────────────────────────────────
    def _classify_by_size(self, bbox: tuple, frame_w: int, frame_h: int) -> str:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        total_area = frame_w * frame_h

        ratio = self._smooth_ratio(area / total_area)

        if ratio < 0.015:
            return "obstáculo menor"
        elif ratio < 0.08:
            return "obstáculo mediano"
        elif ratio < 0.30:
            return "obstáculo grande"
        else:
            return "barrera"

    # ─────────────────────────────────────────────
    def estimate_dimensions(self, bbox: tuple, frame_w: int, frame_h: int) -> str:
        x1, y1, x2, y2 = bbox
        w_px = x2 - x1
        h_px = y2 - y1

        w_pct = round(w_px / frame_w * 100, 1)
        h_pct = round(h_px / frame_h * 100, 1)

        return f"{w_px}×{h_px}px ({w_pct}%×{h_pct}%)"

    # ─────────────────────────────────────────────
    def estimate_distance(self, bbox: tuple, frame_h: int) -> str:
        x1, y1, x2, y2 = bbox
        h = y2 - y1

        if h < 10:
            return "lejos"

        # más estable que 1000/h
        distance = 2.5 * (frame_h / h)

        return f"{round(distance, 2)}m"