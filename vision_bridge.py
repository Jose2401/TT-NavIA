"""
vision_bridge.py
Puente entre el módulo de visión REAL del proyecto (vision/TT-NavIA, basado
en YOLOv8-seg) y el SLAM.

Responsabilidades:
1. Ejecutar el detector real (ObjectDetector) y el rastreo de movimiento
   por objeto (ObjectMotionTracker) sobre cada frame de cámara.
2. Convertir cada detección (bbox + clase) en una VisionDetection para el
   SLAM: bearing a partir de la columna central del bbox y el FOV de la
   cámara, y una distancia monocular gruesa a partir de la altura del bbox
   (modelo pinhole con alturas típicas por clase).
3. Marcar/filtrar objetos dinámicos (personas, mascotas, vehículos, o
   cualquier objeto que el tracker vea moviéndose): un objeto que se mueve
   NO puede ser landmark, corrompería la localización.
4. Fusionar las detecciones con las lecturas ultrasónicas
   (fuse_observations): el ultrasonido aporta el rango preciso cuando
   apunta hacia la detección; si no hay ultrasonido en esa dirección se
   usa la distancia monocular con una incertidumbre mucho mayor
   (sigma_r proporcional), y el EKF pondera cada fuente correctamente.

La fusión (fuse_observations / free_space_rays) es puro cálculo y no
depende de la cámara: sim_world.py la reutiliza con datos simulados.
"""
import math
import os
import sys

import numpy as np

from interfaces import VisionDetection, Observation

# El paquete de visión vive en vision/TT-NavIA (nombre de carpeta con guion,
# no importable como paquete normal): se agrega al path explícitamente.
TTNAVIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vision", "TT-NavIA")

# Clases inherentemente dinámicas: aunque en un frame estén quietas, no son
# parte estable del cuarto y no deben volverse landmarks.
DYNAMIC_LABELS = {
    "person", "dog", "cat", "bird", "horse", "sheep", "cow",
    "bicycle", "motorcycle", "car", "truck", "bus",
}

# Altura real típica (m) por clase, para la distancia monocular pinhole:
# distancia = f_px * altura_real / altura_bbox_px. Para clases fuera de la
# tabla se usa la heurística original del módulo de visión (2.5 * H / h).
TYPICAL_HEIGHTS_M = {
    "person": 1.65, "chair": 0.90, "table": 0.75, "sofa": 0.85,
    "bed": 0.60, "monitor": 0.45, "potted plant": 0.60, "fridge": 1.70,
    "toilet": 0.75, "sink": 0.85, "backpack": 0.45, "suitcase": 0.60,
    "bottle": 0.25, "cup": 0.10, "laptop": 0.25, "oven": 0.75,
    "dog": 0.50, "cat": 0.30, "door": 2.00,
}

# Incertidumbre por fuente de rango (desv. estándar):
SIGMA_R_ULTRASONIC = 0.05          # m — el ultrasonido es preciso
SIGMA_R_MONOCULAR_PCT = 0.30       # fracción del rango — monocular es grueso
SIGMA_PHI_VISION = math.radians(3)     # la cámara es buena en dirección
SIGMA_PHI_ULTRASONIC = math.radians(10)  # cono ancho del sensor ultrasónico


class VisionBridge:
    """Envuelve el detector real de vision/TT-NavIA y produce
    VisionDetection listas para el SLAM."""

    def __init__(self, model_path=None, confidence=0.45,
                 hfov_deg=70.0, vfov_deg=55.0, infer_size=(640, 480)):
        if TTNAVIA_DIR not in sys.path:
            sys.path.insert(0, TTNAVIA_DIR)
        # Imports diferidos: ultralytics/torch tardan en cargar y solo se
        # necesitan si realmente se usa la cámara.
        from detector import ObjectDetector
        from obstacle_logic import ObstacleClassifier, LABEL_ALIASES
        from motion import ObjectMotionTracker

        if model_path is None:
            model_path = os.path.join(TTNAVIA_DIR, "yolov8s-seg.pt")

        self.detector = ObjectDetector(model_path=model_path, confidence=confidence)
        self.classifier = ObstacleClassifier()
        self.tracker = ObjectMotionTracker()
        self.aliases = LABEL_ALIASES

        self.hfov = math.radians(hfov_deg)
        self.vfov = math.radians(vfov_deg)
        self.infer_size = infer_size

    def process(self, frame_bgr):
        """Ejecuta el detector sobre un frame BGR (como lo entrega OpenCV)
        y devuelve la lista de VisionDetection (incluye los objetos en
        movimiento, ya marcados con moving=True para que la fusión los
        excluya de los landmarks)."""
        import cv2
        frame = cv2.resize(frame_bgr, self.infer_size, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        f_px = 0.5 * h / math.tan(self.vfov / 2.0)   # focal en píxeles

        raw = self.detector.detect(frame)
        moving_flags = self.tracker.update(raw)

        detections = []
        for det, is_moving in zip(raw, moving_flags):
            x1, y1, x2, y2 = det["bbox"]
            label = self.aliases.get(det["label"].lower(), det["label"].lower())
            cx = (x1 + x2) / 2.0
            bbox_h = max(1, y2 - y1)

            # Columna 0 = borde izquierdo de la imagen -> bearing positivo
            bearing = ((w / 2.0 - cx) / w) * self.hfov

            if label in TYPICAL_HEIGHTS_M:
                range_est = f_px * TYPICAL_HEIGHTS_M[label] / bbox_h
            else:
                range_est = 2.5 * (h / bbox_h)   # heurística de obstacle_logic
            range_est = max(0.2, min(range_est, 8.0))

            detections.append(VisionDetection(
                bearing=bearing,
                label=label,
                pixel_width=x2 - x1,
                range_est=range_est,
                moving=is_moving or label in DYNAMIC_LABELS,
                confidence=det["confidence"],
            ))
        return detections


class VisualYawEstimator:
    """Estima el GIRO (delta de yaw/theta) entre frames consecutivos por
    flujo óptico, para que el SLAM sepa "hacia dónde voltea" el usuario
    usando SOLO la cámara (sin IMU ni odometría).

    Método: rastrear esquinas (Shi-Tomasi + Lucas-Kanade piramidal) entre
    el frame anterior y el actual; la MEDIANA del desplazamiento
    horizontal de los puntos rastreados se convierte a ángulo con la
    focal en píxeles: delta_yaw = atan(dx_mediana / f). La mediana lo hace
    robusto a objetos en movimiento (una persona cruzando arrastra solo a
    una minoría de los puntos).

    Convención de signos: girar a la IZQUIERDA (+theta) hace que la escena
    se desplace a la DERECHA en la imagen (+dx) -> delta positivo,
    consistente con el bearing (+ = izquierda) del resto del sistema.

    Limitación (deliberada): NO estima traslación — la escala monocular es
    ambigua sin más sensores. La posición (x, y) la corrige el EKF con las
    distancias monoculares a los objetos detectados, y mejorará cuando
    exista el módulo real de odometría/IMU."""

    def __init__(self, hfov_deg=70.0, work_width=640,
                 max_features=200, min_features=12):
        self.hfov = math.radians(hfov_deg)
        self.work_width = work_width
        self.max_features = max_features
        self.min_features = min_features
        self.prev_gray = None

    def update(self, frame_bgr):
        """Devuelve (delta_rot_rad, tracking_ok). Si no hay suficiente
        textura para rastrear, devuelve (0.0, False): el SLAM continúa en
        dead-reckoning y su confianza lo reflejará."""
        import cv2
        h, w = frame_bgr.shape[:2]
        scale = self.work_width / float(w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.work_width, int(h * scale)))

        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return 0.0, False

        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_features,
                                      qualityLevel=0.01, minDistance=12)
        if pts is None or len(pts) < self.min_features:
            return 0.0, False

        nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = status.ravel() == 1
        if good.sum() < self.min_features:
            return 0.0, False

        dx = (nxt[good, 0, 0] - pts[good, 0, 0])
        f_px = (self.work_width / 2.0) / math.tan(self.hfov / 2.0)
        delta_yaw = math.atan(float(np.median(dx)) / f_px)
        return delta_yaw, True


# ----------------------------------------------------------------------
# Fusión visión + ultrasonido (independiente de la cámara)
# ----------------------------------------------------------------------
def fuse_observations(vision_dets, ultrasonic_readings,
                      bearing_tol=math.radians(20)):
    """Construye las observaciones (rango, bearing) que entran al EKF:

    - Detección visual + eco ultrasónico en la misma dirección Y con rango
      compatible con la estimación monocular -> rango del ultrasonido
      (preciso) con el bearing de visión (preciso). La mejor observación
      posible. La verificación de compatibilidad evita fusionar el bearing
      de un objeto con el eco de OTRO objeto/pared que casualmente cae en
      el cono del sensor (eso crearía landmarks fantasma).
    - Detección visual sin ultrasonido compatible -> rango monocular con
      sigma_r grande (30% del rango): sigue corrigiendo, pero pesa menos.
    - Eco ultrasónico sin detección visual asociada (pared lisa, objeto sin
      clase YOLO) -> observación con el bearing del sensor y sigma_phi
      ancho (cono del sensor). Puede corregir contra landmarks ya
      conocidos, pero NO crea landmarks (can_init_landmark=False): el
      punto de eco de una pared se desliza al caminar y no es un punto
      fijo del mundo. Para el mapa de ocupación sí se usa siempre.
    - Los objetos en movimiento se descartan: no son landmarks.
    """
    observations = []
    valid_us = [u for u in ultrasonic_readings if u.valid]
    used_us = set()

    for det in vision_dets:
        if det.moving:
            continue
        best_i, best_diff = None, bearing_tol
        for i, u in enumerate(valid_us):
            diff = abs(u.sensor_bearing - det.bearing)
            if diff < best_diff and _range_compatible(u.range_m, det.range_est):
                best_i, best_diff = i, diff
        if best_i is not None:
            u = valid_us[best_i]
            used_us.add(best_i)
            observations.append(Observation(
                range_m=u.range_m, bearing=det.bearing, label=det.label,
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_VISION))
        elif det.range_est is not None:
            observations.append(Observation(
                range_m=det.range_est, bearing=det.bearing, label=det.label,
                sigma_r=SIGMA_R_MONOCULAR_PCT * det.range_est,
                sigma_phi=SIGMA_PHI_VISION,
                # Sin respaldo ultrasónico se exige más evidencia antes de
                # confirmar el landmark: a framerate de cámara un falso
                # positivo de YOLO dura 1-2 frames y no debe aparecer en
                # el mapa como obstáculo "inventado".
                min_confirm_hits=4))

    for i, u in enumerate(valid_us):
        if i not in used_us:
            observations.append(Observation(
                range_m=u.range_m, bearing=u.sensor_bearing, label="eco_us",
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_ULTRASONIC,
                min_confirm_hits=3))

    return observations


def _range_compatible(us_range, mono_range):
    """True si el rango ultrasónico es plausible para una detección cuya
    distancia monocular estimada es mono_range. La estimación monocular es
    gruesa (~30%), así que la tolerancia es generosa; su función es solo
    descartar emparejamientos absurdos (eco de la pared del fondo asignado
    a una silla cercana, o viceversa)."""
    if mono_range is None:
        return True
    return abs(us_range - mono_range) <= max(0.6, 0.5 * mono_range)


def free_space_rays(ultrasonic_readings):
    """Rayos "sin eco" (el sensor no midió nada dentro de su alcance): no
    sirven para corregir la pose, pero sí informan al mapa de ocupación que
    ese corredor está libre. Se entregan SOLO a OccupancyGrid, nunca al
    EKF (range_m > max_range hace que el Ray Casting no marque celda
    ocupada al final del rayo)."""
    return [
        Observation(range_m=u.max_range + 0.01, bearing=u.sensor_bearing,
                    label="libre")
        for u in ultrasonic_readings if not u.valid
    ]
