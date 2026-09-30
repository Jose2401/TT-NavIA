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
SIGMA_R_GROUND_PLANE_PCT = 0.20    # proyección al plano del piso (mejor que
                                   # el pinhole por altura, peor que el US)
SIGMA_PHI_VISION = math.radians(3)     # la cámara es buena en dirección
SIGMA_PHI_ULTRASONIC = math.radians(10)  # cono ancho del sensor ultrasónico

# Perfiles de cámara (hfov, vfov en grados). El hardware final es la
# Raspberry Pi Camera Module 3 Wide (IMX708, FOV diagonal 120°,
# ~102° H x 67° V según la documentación del proyecto); en la laptop se
# prueba con la webcam (~70° H típico).
CAMERA_PROFILES = {
    "laptop": {"hfov_deg": 70.0, "vfov_deg": 55.0},
    "picam3wide": {"hfov_deg": 102.0, "vfov_deg": 67.0},
}

# Geometría del montaje: la cámara va en los lentes del usuario.
DEFAULT_CAM_HEIGHT_M = 1.55    # altura de los ojos de un adulto promedio
DEFAULT_CAM_PITCH_DEG = 10.0   # inclinación hacia abajo al caminar mirando
                               # al frente (ajustable por calibración)


def ground_distance_from_row(row, img_h, vfov_rad,
                             cam_height_m=DEFAULT_CAM_HEIGHT_M,
                             cam_pitch_deg=DEFAULT_CAM_PITCH_DEG):
    """Distancia horizontal (m) al punto del PISO que se proyecta en la
    fila `row` de la imagen, asumiendo piso plano y cámara a
    `cam_height_m` con inclinación `cam_pitch_deg` hacia abajo.

    Es la única forma monocular de medir distancia a cosas PLANAS en el
    suelo (hoyos, coladeras, charcos): el modelo pinhole por altura de
    bbox no aplica porque su "altura" aparente no corresponde a una
    altura física. También sirve como segunda estimación para objetos
    apoyados en el piso (la base del bbox toca el suelo).

    Devuelve None si la fila queda en o sobre el horizonte (no es piso).
    """
    f_px = 0.5 * img_h / math.tan(vfov_rad / 2.0)
    # Ángulo bajo la horizontal del rayo que pasa por esa fila
    phi = math.radians(cam_pitch_deg) + math.atan((row - img_h / 2.0) / f_px)
    if phi < math.radians(2.0):        # horizonte o por encima: sin piso
        return None
    return cam_height_m / math.tan(phi)


class VisionBridge:
    """Envuelve el detector real de vision/TT-NavIA y produce
    VisionDetection listas para el SLAM."""

    def __init__(self, model_path=None, confidence=0.45,
                 hfov_deg=None, vfov_deg=None, infer_size=(640, 480),
                 camera_profile="laptop",
                 cam_height_m=DEFAULT_CAM_HEIGHT_M,
                 cam_pitch_deg=DEFAULT_CAM_PITCH_DEG,
                 use_ground_plane=True):
        """
        camera_profile: 'laptop' (webcam ~70°) o 'picam3wide' (Raspberry
            Pi Camera Module 3 Wide, el hardware final). hfov_deg /
            vfov_deg explícitos tienen prioridad sobre el perfil.
        cam_height_m / cam_pitch_deg: montaje de la cámara (lentes del
            usuario), para la distancia por proyección al plano del piso.
        use_ground_plane: si True, la distancia monocular de los objetos
            apoyados en el piso se refina con la fila donde su bbox toca
            el suelo (suele ser más estable que el pinhole por altura,
            cuya altura "típica" por clase es solo un promedio).
        """
        profile = CAMERA_PROFILES.get(camera_profile,
                                      CAMERA_PROFILES["laptop"])
        hfov_deg = hfov_deg if hfov_deg is not None else profile["hfov_deg"]
        vfov_deg = vfov_deg if vfov_deg is not None else profile["vfov_deg"]

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
        self.cam_height_m = cam_height_m
        self.cam_pitch_deg = cam_pitch_deg
        self.use_ground_plane = use_ground_plane

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

            # Refinamiento por plano de piso: si la base del bbox está
            # dentro del frame (el objeto no está cortado por el borde),
            # la fila donde toca el suelo da otra estimación de
            # distancia, independiente de la altura "típica" de la
            # clase. Se limita a ±60% de la estimación por altura para
            # que un pitch mal calibrado no la arrastre.
            if self.use_ground_plane and y2 < h * 0.97:
                gd = ground_distance_from_row(
                    y2, h, self.vfov, self.cam_height_m,
                    self.cam_pitch_deg)
                if gd is not None:
                    lo, hi = 0.4 * range_est, 1.6 * range_est
                    range_est = min(hi, max(lo, gd))

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


class GroundHazardDetector:
    """Detecta peligros A NIVEL DE PISO que YOLO no puede ver porque no
    son "objetos" COCO: hoyos, coladeras/alcantarillas, charcos/agua y
    escaleras/escalones. Es la respuesta a la limitación declarada en la
    documentación del proyecto ("no detecta obstáculos a nivel de suelo").

    Método: la segmentación de escena ADE20K (SegFormer, ya presente en
    vision/TT-NavIA) clasifica cada pixel. Sobre ella:
      1. AGUA y ESCALERAS son clases directas de ADE20K -> peligro.
      2. HOYOS/COLADERAS: una región que NO es superficie transitable
         pero está RODEADA de superficie transitable en la mitad baja de
         la imagen es un agujero/anomalía del piso. (Una coladera abierta
         es exactamente eso: un parche no-piso dentro del piso.)
    La distancia se mide por proyección al plano del piso
    (ground_distance_from_row): el pinhole por altura de bbox no aplica a
    cosas planas.

    Coste: SegFormer-B0 es el paso más caro después de YOLO; en la
    Raspberry conviene ejecutarlo cada N frames (parámetro del main).
    Los peligros van al mapa con is_hazard=True y el EKF les exige
    varios avistamientos (min_confirm_hits) antes de confirmar, así que
    la baja frecuencia no inventa peligros falsos.
    """

    # Área mínima/máxima del componente como fracción del frame (filtra
    # ruido de segmentación y evita marcar media imagen como "hoyo").
    MIN_AREA_FRAC = 0.0015
    MAX_AREA_FRAC = 0.20
    # Fracción mínima del borde del componente que debe ser piso
    # transitable para considerarlo "agujero EN el piso".
    MIN_SURROUND = 0.55

    def __init__(self, hfov_deg=None, vfov_deg=None,
                 camera_profile="laptop",
                 cam_height_m=DEFAULT_CAM_HEIGHT_M,
                 cam_pitch_deg=DEFAULT_CAM_PITCH_DEG,
                 work_size=(480, 360), max_range_m=6.0):
        profile = CAMERA_PROFILES.get(camera_profile,
                                      CAMERA_PROFILES["laptop"])
        self.hfov = math.radians(hfov_deg if hfov_deg is not None
                                 else profile["hfov_deg"])
        self.vfov = math.radians(vfov_deg if vfov_deg is not None
                                 else profile["vfov_deg"])
        self.cam_height_m = cam_height_m
        self.cam_pitch_deg = cam_pitch_deg
        self.work_size = work_size
        self.max_range_m = max_range_m
        self._segmenter = None      # carga diferida (torch/transformers)

    def _ensure_model(self):
        if self._segmenter is None:
            if TTNAVIA_DIR not in sys.path:
                sys.path.insert(0, TTNAVIA_DIR)
            from scene_segmentation import SceneSegmenter
            self._segmenter = SceneSegmenter()

    def process(self, frame_bgr, yolo_detections=None):
        """Devuelve una lista de VisionDetection con hazard=True.
        yolo_detections (opcional): detecciones del mismo frame; un
        "agujero" que se solapa con un objeto YOLO no es un hoyo, es la
        base del objeto, y se descarta."""
        import cv2
        self._ensure_model()

        small = cv2.resize(frame_bgr, self.work_size,
                           interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        seg = self._segmenter.segment(small)
        walkable, water, stairs = self._segmenter.extract_ground_masks(seg)

        hazards = []
        hazards += self._mask_hazards(water, "agua", w, h)
        hazards += self._mask_hazards(stairs, "escalera", w, h)
        hazards += self._hole_hazards(walkable, water, stairs, w, h)

        if yolo_detections:
            hazards = [hz for hz in hazards
                       if not self._overlaps_detection(hz, yolo_detections)]
        return hazards

    # ----------------- internos -----------------
    def _component_detection(self, comp_mask, label, w, h):
        """Convierte un componente conexo de la máscara en una
        VisionDetection hazard, con bearing por centroide y distancia por
        plano de piso en la fila MÁS CERCANA al usuario (la fila inferior
        del componente: el borde del hoyo más próximo es el que importa
        para alertar)."""
        ys, xs = np.nonzero(comp_mask)
        if len(xs) == 0:
            return None
        cx = float(xs.mean())
        bottom_row = float(ys.max())
        dist = ground_distance_from_row(
            bottom_row, h, self.vfov, self.cam_height_m, self.cam_pitch_deg)
        if dist is None or dist > self.max_range_m:
            return None
        bearing = ((w / 2.0 - cx) / w) * self.hfov
        return VisionDetection(
            bearing=bearing, label=label,
            pixel_width=int(xs.max() - xs.min()),
            range_est=max(0.2, dist), moving=False,
            confidence=0.6, hazard=True)

    def _mask_hazards(self, mask, label, w, h):
        """Componentes conexos de una máscara de clase directa (agua,
        escalera) en la mitad baja de la imagen."""
        import cv2
        mask = mask.copy()
        mask[: int(h * 0.45), :] = 0        # lejos del horizonte no es piso
        n, labels_img = cv2.connectedComponents(mask.astype(np.uint8))
        out = []
        area = w * h
        for i in range(1, n):
            comp = labels_img == i
            frac = comp.sum() / area
            if frac < self.MIN_AREA_FRAC:
                continue
            det = self._component_detection(comp, label, w, h)
            if det is not None:
                out.append(det)
        return out

    def _hole_hazards(self, walkable, water, stairs, w, h):
        """Agujeros: componentes NO transitables rodeados de piso
        transitable en la zona baja de la imagen. El agua y las
        escaleras ya se reportaron con su propia etiqueta y se excluyen
        (si no, cada charco saldría duplicado como 'hoyo')."""
        import cv2
        area = w * h
        # Solo interesa la franja donde el piso es visible
        band = np.zeros_like(walkable)
        band[int(h * 0.45):, :] = 1
        not_walk = ((walkable == 0) & (water == 0) & (stairs == 0)
                    & (band == 1)).astype(np.uint8)

        n, labels_img = cv2.connectedComponents(not_walk)
        out = []
        kernel = np.ones((5, 5), np.uint8)
        for i in range(1, n):
            comp = (labels_img == i)
            frac = comp.sum() / area
            if frac < self.MIN_AREA_FRAC or frac > self.MAX_AREA_FRAC:
                continue
            ys, xs = np.nonzero(comp)
            # Un componente pegado a los bordes laterales/inferior del
            # RECORTE suele ser continuación de algo grande (mueble,
            # pared, el propio cuerpo), no un agujero aislado.
            if xs.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
                continue
            # ¿Rodeado de piso? El anillo alrededor del componente debe
            # ser mayoritariamente transitable.
            comp_u8 = comp.astype(np.uint8)
            ring = cv2.dilate(comp_u8, kernel) - comp_u8
            ring_px = ring.sum()
            if ring_px == 0:
                continue
            surround = float((walkable[ring == 1]).sum()) / ring_px
            if surround < self.MIN_SURROUND:
                continue
            det = self._component_detection(comp, "hoyo", w, h)
            if det is not None:
                out.append(det)
        return out

    def _overlaps_detection(self, hazard, detections,
                            tol=math.radians(6)):
        """True si el peligro apunta en la misma dirección que un objeto
        YOLO a distancia similar (entonces es la base del objeto)."""
        for det in detections:
            if det.hazard:
                continue
            if abs(det.bearing - hazard.bearing) < tol and (
                    det.range_est is None or hazard.range_est is None
                    or abs(det.range_est - hazard.range_est)
                    < max(0.8, 0.4 * det.range_est)):
                return True
        return False


# ----------------------------------------------------------------------
# Nivel de riesgo por detección (tabla de riesgo del proyecto, §7.1):
# 0 libre, 1 bajo, 2 medio, 3 alto. Es el r_* del vector de estado del
# módulo de navegación (DRL).
# ----------------------------------------------------------------------
HIGH_RISK_LABELS = {"escalera", "hoyo", "agua", "stairs"}


def detection_risk(det: VisionDetection):
    """Riesgo 0-3 de una detección según la tabla del proyecto:
    persona/objeto EN MOVIMIENTO = alto; escalera/desnivel/hoyo = alto;
    obstáculo estático = medio; pared/superficie = bajo."""
    if det.hazard or det.label in HIGH_RISK_LABELS:
        return 3
    if det.moving:
        return 3
    if det.label in ("wall", "eco_us", "libre"):
        return 1
    return 2


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
        if det.hazard:
            # Peligro de piso: el ultrasonido NO lo ve (no sobresale),
            # así que nunca se empareja con un eco. Va al EKF como
            # observación normal (una coladera es un punto fijo útil
            # para corregir) y al mapa con is_hazard=True para la capa
            # de peligros. min_confirm_hits alto: un parpadeo de la
            # segmentación no debe pintar un hoyo inexistente.
            if det.range_est is not None:
                observations.append(Observation(
                    range_m=det.range_est, bearing=det.bearing,
                    label=det.label,
                    sigma_r=SIGMA_R_GROUND_PLANE_PCT * det.range_est,
                    sigma_phi=SIGMA_PHI_VISION,
                    min_confirm_hits=4, is_hazard=True))
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
            # Eco sin respaldo visual: SOLO alimenta al mapa de
            # ocupación (can_correct=False). Su punto de reflexión se
            # desliza por la pared al caminar; usarlo como landmark
            # puntual corrompía la pose (ATE 0.53 -> 0.27 al excluirlo,
            # 6 semillas, geometría del simulador ya corregida).
            observations.append(Observation(
                range_m=u.range_m, bearing=u.sensor_bearing, label="eco_us",
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_ULTRASONIC,
                can_init_landmark=False, can_correct=False))

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
