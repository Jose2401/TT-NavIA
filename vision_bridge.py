"""
vision_bridge.py — puente entre el modulo de vision (yolo + segformer)
y el slam. convierte cada bbox en (bearing, distancia monocular),
marca lo que se mueve pa que no se vuelva landmark, y fusiona con el
ultrasonido. la fusion es puro calculo, sim_world la reusa sin camara.
"""
import math
import os

import numpy as np

from interfaces import VisionDetection, Observation

VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vision")

# cosas que se mueven solas: nunca landmarks, corromperian la pose
DYNAMIC_LABELS = {
    "person", "dog", "cat", "bird", "horse", "sheep", "cow",
    "bicycle", "motorcycle", "car", "truck", "bus",
}

# alturas tipicas (m) para el pinhole: dist = f_px * H_real / h_bbox
TYPICAL_HEIGHTS_M = {
    "person": 1.65, "chair": 0.90, "table": 0.75, "sofa": 0.85,
    "bed": 0.60, "monitor": 0.45, "potted plant": 0.60, "fridge": 1.70,
    "toilet": 0.75, "sink": 0.85, "backpack": 0.45, "suitcase": 0.60,
    "bottle": 0.25, "cup": 0.10, "laptop": 0.25, "oven": 0.75,
    "dog": 0.50, "cat": 0.30, "door": 2.00,
}

# sigmas por fuente: el ultrasonido mide rango bien chido (5 cm), el
# monocular es grueso (~30%), el plano de piso queda en medio
SIGMA_R_ULTRASONIC = 0.05
SIGMA_R_MONOCULAR_PCT = 0.30
SIGMA_R_GROUND_PLANE_PCT = 0.20
SIGMA_PHI_VISION = math.radians(3)       # la camara apunta bien
SIGMA_PHI_ULTRASONIC = math.radians(10)  # cono ancho del sensor

# hardware final: picam 3 wide (~102x67 grados); en la laptop, webcam
CAMERA_PROFILES = {
    "laptop": {"hfov_deg": 70.0, "vfov_deg": 55.0},
    "picam3wide": {"hfov_deg": 102.0, "vfov_deg": 67.0},
}

# la camara va montada en los lentes del usuario
DEFAULT_CAM_HEIGHT_M = 1.55
DEFAULT_CAM_PITCH_DEG = 10.0


def ground_distance_from_row(row, img_h, vfov_rad,
                             cam_height_m=DEFAULT_CAM_HEIGHT_M,
                             cam_pitch_deg=DEFAULT_CAM_PITCH_DEG):
    """distancia al punto del piso que cae en la fila `row` (piso plano).
    es la unica forma monocular de medir cosas PLANAS (hoyos, charcos):
    el pinhole por altura de bbox no sirve pq no tienen altura real.
    None si la fila queda en o sobre el horizonte."""
    f_px = 0.5 * img_h / math.tan(vfov_rad / 2.0)
    phi = math.radians(cam_pitch_deg) + math.atan((row - img_h / 2.0) / f_px)
    if phi < math.radians(2.0):
        return None
    return cam_height_m / math.tan(phi)


class VisionBridge:
    """envuelve el detector real y saca VisionDetection para el slam."""

    def __init__(self, model_path=None, confidence=0.45,
                 hfov_deg=None, vfov_deg=None, infer_size=(640, 480),
                 camera_profile="laptop",
                 cam_height_m=DEFAULT_CAM_HEIGHT_M,
                 cam_pitch_deg=DEFAULT_CAM_PITCH_DEG,
                 use_ground_plane=True):
        profile = CAMERA_PROFILES.get(camera_profile,
                                      CAMERA_PROFILES["laptop"])
        hfov_deg = hfov_deg if hfov_deg is not None else profile["hfov_deg"]
        vfov_deg = vfov_deg if vfov_deg is not None else profile["vfov_deg"]

        # imports diferidos, torch tarda un buen en cargar
        from vision.detector import ObjectDetector
        from vision.obstacle_logic import ObstacleClassifier, LABEL_ALIASES
        from vision.motion import ObjectMotionTracker

        if model_path is None:
            model_path = os.path.join(VISION_DIR, "yolov8s-seg.pt")

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

    def process(self, frame_bgr, ego_yaw=0.0):
        """detecta sobre un frame bgr. ego_yaw compensa el giro de
        cabeza: si no se resta, al girar TODO sale como 'movil'."""
        import cv2
        frame = cv2.resize(frame_bgr, self.infer_size, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        f_px = 0.5 * h / math.tan(self.vfov / 2.0)

        raw = self.detector.detect(frame)
        self._compensate_ego_motion(ego_yaw, w)
        moving_flags = self.tracker.update(raw)

        detections = []
        for det, is_moving in zip(raw, moving_flags):
            x1, y1, x2, y2 = det["bbox"]
            label = self.aliases.get(det["label"].lower(), det["label"].lower())
            cx = (x1 + x2) / 2.0
            bbox_h = max(1, y2 - y1)

            # columna 0 = borde izquierdo -> bearing positivo
            bearing = ((w / 2.0 - cx) / w) * self.hfov

            if label in TYPICAL_HEIGHTS_M:
                range_est = f_px * TYPICAL_HEIGHTS_M[label] / bbox_h
            else:
                range_est = 2.5 * (h / bbox_h)

            # si la base del bbox toca piso dentro del frame, el plano de
            # piso da otra estimacion; se acota pa que un pitch chueco
            # no la arrastre
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

    def _compensate_ego_motion(self, ego_yaw, img_w):
        shift_tracker_for_yaw(self.tracker, ego_yaw, self.hfov, img_w)


def shift_tracker_for_yaw(tracker, ego_yaw, hfov, img_w):
    """corre el historial del tracker los pixeles que la escena estatica
    se movio por el giro (girar a la izq lleva cx a cx + yaw*w/hfov)."""
    if not ego_yaw or hfov <= 0:
        return 0.0
    dx = ego_yaw * img_w / hfov
    if abs(dx) < 0.5:
        return 0.0
    tracker.prev_bboxes = [(x1 + dx, y1, x2 + dx, y2)
                           for (x1, y1, x2, y2) in tracker.prev_bboxes]
    for dq in tracker.history.values():
        shifted = [(cx + dx, cy) for (cx, cy) in dq]
        dq.clear()
        dq.extend(shifted)
    return dx


class VisualYawEstimator:
    """delta de yaw entre frames por flujo optico (shi-tomasi + LK,
    mediana del dx pa aguantar gente cruzando). girar a la IZQ (+theta)
    mueve la escena a la DERECHA (+dx). no estima traslacion: la escala
    monocular es ambigua, de eso se encarga el ekf."""

    def __init__(self, hfov_deg=70.0, work_width=640,
                 max_features=200, min_features=12):
        self.hfov = math.radians(hfov_deg)
        self.work_width = work_width
        self.max_features = max_features
        self.min_features = min_features
        self.prev_gray = None

    def update(self, frame_bgr):
        """-> (delta_rot_rad, tracking_ok). sin textura da (0.0, False)."""
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
    """peligros a nivel de piso que yolo no ve (hoyos, coladeras, agua,
    escaleras). agua/escaleras son clases directas de ade20k; un hoyo es
    un parche NO transitable rodeado de piso transitable. la distancia
    sale por plano de piso pq el pinhole no aplica a cosas planas.
    segformer es caro: en la pi correrlo cada N frames esta bien, el ekf
    pide varios avistamientos antes de confirmar."""

    # area min/max del componente (filtra ruido y media-imagen-hoyo xd)
    MIN_AREA_FRAC = 0.0015
    MAX_AREA_FRAC = 0.20
    # fraccion del anillo que debe ser piso pa contar como hoyo
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
            from vision.scene_segmentation import SceneSegmenter
            self._segmenter = SceneSegmenter()

    def process(self, frame_bgr, yolo_detections=None):
        """lista de VisionDetection con hazard=True. un 'hoyo' que se
        solapa con un objeto yolo es la base del objeto, se tira."""
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
        """componente conexo -> VisionDetection; la distancia se mide en
        la fila de abajo del componente (el borde cercano del hoyo)."""
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
        """componentes de una clase directa (agua, escalera) abajo."""
        import cv2
        mask = mask.copy()
        mask[: int(h * 0.45), :] = 0        # cerca del horizonte no es piso
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
        """hoyos: no-transitable rodeado de piso. agua/escaleras ya
        salieron con su etiqueta, se excluyen pa no duplicar."""
        import cv2
        area = w * h
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
            # pegado al borde = continuacion de algo grande, no un hoyo
            if xs.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
                continue
            # el anillo alrededor debe ser mayormente piso
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
        """True si apunta igual que un objeto yolo a distancia similar."""
        for det in detections:
            if det.hazard:
                continue
            if abs(det.bearing - hazard.bearing) < tol and (
                    det.range_est is None or hazard.range_est is None
                    or abs(det.range_est - hazard.range_est)
                    < max(0.8, 0.4 * det.range_est)):
                return True
        return False


# riesgo 0-3 por deteccion (tabla del proyecto)
HIGH_RISK_LABELS = {"escalera", "hoyo", "agua", "stairs"}


def detection_risk(det: VisionDetection):
    """riesgo 0-3: movil o peligro de piso = 3, estatico = 2, pared = 1."""
    if det.hazard or det.label in HIGH_RISK_LABELS:
        return 3
    if det.moving:
        return 3
    if det.label in ("wall", "eco_us", "libre"):
        return 1
    return 2


# ----------------------------------------------------------------------
# fusion vision + ultrasonido (no depende de la camara)
# ----------------------------------------------------------------------
def fuse_observations(vision_dets, ultrasonic_readings,
                      bearing_tol=math.radians(20)):
    """arma las observaciones del ekf:
    - vision + eco compatible: rango del us (preciso) + bearing de vision.
      el check de compatibilidad evita casar el bearing de un objeto con
      el eco de OTRA cosa (landmarks fantasma).
    - vision sola: rango monocular con sigma gorda (30%), pesa menos.
    - eco sin respaldo visual: solo mapa de ocupacion (can_correct=False),
      el punto de eco se desliza por la pared al caminar y usarlo de
      landmark corrompia la pose (ATE 0.53 -> 0.27 al quitarlo).
    - lo que se mueve se tira."""
    observations = []
    valid_us = [u for u in ultrasonic_readings if u.valid]
    used_us = set()

    for det in vision_dets:
        if det.moving:
            continue
        if det.hazard:
            # el ultrasonido no ve hoyos, no lo emparejes con esto.
            # min_confirm_hits alto: un parpadeo de la segmentacion no
            # debe pintar un hoyo inexistente
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
                # sin us se pide mas evidencia: un falso positivo de yolo
                # dura 1-2 frames y no debe entrar al mapa
                min_confirm_hits=4))

    for i, u in enumerate(valid_us):
        if i not in used_us:
            # eco solo: alimenta el mapa pero no corrige ni crea landmarks
            observations.append(Observation(
                range_m=u.range_m, bearing=u.sensor_bearing, label="eco_us",
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_ULTRASONIC,
                can_init_landmark=False, can_correct=False))

    return observations


def _range_compatible(us_range, mono_range):
    """tolerancia generosa (el monocular es grueso); solo descarta
    emparejamientos absurdos tipo eco del fondo con silla cercana."""
    if mono_range is None:
        return True
    return abs(us_range - mono_range) <= max(0.6, 0.5 * mono_range)


def free_space_rays(ultrasonic_readings):
    """rayos sin eco: no corrigen pose pero marcan corredor libre en el
    mapa de ocupacion. van SOLO a OccupancyGrid, nunca al ekf."""
    return [
        Observation(range_m=u.max_range + 0.01, bearing=u.sensor_bearing,
                    label="libre")
        for u in ultrasonic_readings if not u.valid
    ]
