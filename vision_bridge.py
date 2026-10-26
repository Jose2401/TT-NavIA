import math
import os

import numpy as np

from interfaces import VisionDetection, Observation

VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vision")

DYNAMIC_LABELS = {
    "person", "dog", "cat", "bird", "horse", "sheep", "cow",
    "bicycle", "motorcycle", "car", "truck", "bus",
}

TYPICAL_HEIGHTS_M = {
    "person": 1.65, "chair": 0.90, "table": 0.75, "sofa": 0.85,
    "bed": 0.60, "monitor": 0.45, "potted plant": 0.60, "fridge": 1.70,
    "toilet": 0.75, "sink": 0.85, "backpack": 0.45, "suitcase": 0.60,
    "bottle": 0.25, "cup": 0.10, "laptop": 0.25, "oven": 0.75,
    "dog": 0.50, "cat": 0.30, "door": 2.00,
    "window": 1.30, "trash can": 0.95, "street light": 4.0,
    "sign": 0.70, "ladder": 2.00, "shelf": 1.70, "tree": 4.0,
    "fountain": 1.20, "bench": 0.80, "fire hydrant": 0.75,
}

# clases EXTRA del detector open images v7 (600 clases): solo las que
# coco no tiene y nos sirven pa navegar. Stairs va como hazard (riesgo
# alto en la tabla del TT)
EXTRA_OIV7 = {
    "Door": "door", "Stairs": "escalera", "Window": "window",
    "Waste container": "trash can", "Street light": "street light",
    "Traffic sign": "sign", "Ladder": "ladder", "Shelf": "shelf",
    "Tree": "tree", "Fountain": "fountain",
}
EXTRA_HAZARD = {"escalera"}

SIGMA_R_ULTRASONIC = 0.05
SIGMA_R_MONOCULAR_PCT = 0.30
SIGMA_R_GROUND_PLANE_PCT = 0.20
SIGMA_PHI_VISION = math.radians(3)
SIGMA_PHI_ULTRASONIC = math.radians(10)

CAMERA_PROFILES = {
    "laptop": {"hfov_deg": 70.0, "vfov_deg": 55.0},
    "picam3wide": {"hfov_deg": 102.0, "vfov_deg": 67.0},
}

DEFAULT_CAM_HEIGHT_M = 1.55
DEFAULT_CAM_PITCH_DEG = 10.0


def groundDistanceFromRow(row, imgH, vfovRad,
                          camHeightM=DEFAULT_CAM_HEIGHT_M,
                          camPitchDeg=DEFAULT_CAM_PITCH_DEG):
    fPx = 0.5 * imgH / math.tan(vfovRad / 2.0)
    phi = math.radians(camPitchDeg) + math.atan((row - imgH / 2.0) / fPx)
    if phi < math.radians(2.0):
        return None
    return camHeightM / math.tan(phi)


class VisionBridge:
    def __init__(self, modelPath=None, confidence=0.45,
                 hfovDeg=None, vfovDeg=None, inferWidth=640,
                 cameraProfile="laptop",
                 camHeightM=DEFAULT_CAM_HEIGHT_M,
                 camPitchDeg=DEFAULT_CAM_PITCH_DEG,
                 useGroundPlane=True,
                 extraModelPath="auto", extraEvery=5, extraConf=0.30):
        """extraModelPath: segundo yolo (open images, 600 clases) pa
        puertas/escaleras/ventanas etc que coco no tiene. "auto" = usa
        vision/yolov8s-oiv7.pt si existe; None = apagado. corre cada
        extraEvery frames pq cuesta."""
        perfil = CAMERA_PROFILES.get(cameraProfile,
                                     CAMERA_PROFILES["laptop"])
        hfovDeg = hfovDeg if hfovDeg is not None else perfil["hfov_deg"]
        vfovDeg = vfovDeg if vfovDeg is not None else perfil["vfov_deg"]

        from vision.detector import ObjectDetector
        from vision.obstacle_logic import ObstacleClassifier, LABEL_ALIASES
        from vision.motion import ObjectMotionTracker

        if modelPath is None:
            modelPath = os.path.join(VISION_DIR, "yolov8s-seg.pt")

        self.detector = ObjectDetector(model_path=modelPath, confidence=confidence)
        self.classifier = ObstacleClassifier()
        self.tracker = ObjectMotionTracker()
        self.aliases = LABEL_ALIASES

        self.extraDetector = None
        self.extraEvery = max(1, extraEvery)
        self._nFrame = 0
        self._extraIds = {}
        if extraModelPath == "auto":
            ruta = os.path.join(VISION_DIR, "yolov8s-oiv7.pt")
            extraModelPath = ruta if os.path.exists(ruta) else None
        if extraModelPath:
            try:
                self.extraDetector = ObjectDetector(
                    model_path=extraModelPath, confidence=extraConf)
                nombres = self.extraDetector.model.names
                self._extraIds = {i: EXTRA_OIV7[n]
                                  for i, n in nombres.items()
                                  if n in EXTRA_OIV7}
            except Exception as e:
                print(f"[VISION] detector extra no disponible ({e})")
                self.extraDetector = None

        self.hfov = math.radians(hfovDeg)
        self.vfov = math.radians(vfovDeg)
        self.inferWidth = inferWidth
        self.camHeightM = camHeightM
        self.camPitchDeg = camPitchDeg
        self.useGroundPlane = useGroundPlane

    def _redimensionar(self, frameBgr):
        # ancho fijo, aspecto intacto: estirar a 4:3 inflaba las alturas
        # de bbox y las distancias monoculares salian ~25% cortas en 16:9
        import cv2
        h, w = frameBgr.shape[:2]
        esc = self.inferWidth / float(w)
        return cv2.resize(frameBgr, (self.inferWidth, int(round(h * esc))),
                          interpolation=cv2.INTER_AREA)

    # bbox de yolo -> bearing + distancia
    def process(self, frameBgr, egoYaw=0.0):
        frame = self._redimensionar(frameBgr)
        h, w = frame.shape[:2]
        # pixeles cuadrados: una sola focal, derivada del hfov
        fPx = 0.5 * w / math.tan(self.hfov / 2.0)
        vfovEff = 2.0 * math.atan(0.5 * h / fPx)
        self._nFrame += 1

        raw = self.detector.detect(frame)
        self._compensarGiro(egoYaw, w)
        flagsMov = self.tracker.update(raw)

        extras = []
        if (self.extraDetector is not None
                and self._nFrame % self.extraEvery == 0):
            extras = self._detectarExtras(frame)

        detecciones = []
        for det, seMueve in zip(raw, flagsMov):
            detecciones.append(self._aDeteccion(
                det, w, h, fPx, vfovEff,
                moving=seMueve
                or det["label"].lower() in DYNAMIC_LABELS))
        for det in extras:
            # sin chocar con lo que coco ya vio ahi mismo
            if any(_iou(det["bbox"], d["bbox"]) > 0.55 for d in raw):
                continue
            label = det["label"]
            detecciones.append(self._aDeteccion(
                det, w, h, fPx, vfovEff, moving=False,
                hazard=label in EXTRA_HAZARD))
        return detecciones

    def _aDeteccion(self, det, w, h, fPx, vfovEff, moving, hazard=False):
        x1, y1, x2, y2 = det["bbox"]
        label = self.aliases.get(det["label"].lower(), det["label"].lower())
        cx = (x1 + x2) / 2.0
        bboxH = max(1, y2 - y1)
        bearing = ((w / 2.0 - cx) / w) * self.hfov

        if label in TYPICAL_HEIGHTS_M:
            rangoEst = fPx * TYPICAL_HEIGHTS_M[label] / bboxH
        else:
            rangoEst = 2.5 * (h / bboxH)

        if self.useGroundPlane and y2 < h * 0.97:
            gd = groundDistanceFromRow(
                y2, h, vfovEff, self.camHeightM, self.camPitchDeg)
            if gd is not None:
                lo, hi = 0.4 * rangoEst, 1.6 * rangoEst
                rangoEst = min(hi, max(lo, gd))

        rangoEst = max(0.2, min(rangoEst, 8.0))
        return VisionDetection(
            bearing=bearing, label=label, pixel_width=x2 - x1,
            range_est=rangoEst, moving=moving,
            confidence=det["confidence"], hazard=hazard)

    def _detectarExtras(self, frame):
        res = self.extraDetector.model(frame, stream=False, verbose=False,
                                       conf=self.extraDetector.confidence)
        out = []
        for r in res:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in self._extraIds:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append({"label": self._extraIds[cls],
                            "confidence": float(box.conf[0]),
                            "bbox": (x1, y1, x2, y2), "mask": None})
        return out

    def _compensarGiro(self, egoYaw, imgW):
        shiftTrackerForYaw(self.tracker, egoYaw, self.hfov, imgW)


def _iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def shiftTrackerForYaw(tracker, egoYaw, hfov, imgW):
    if not egoYaw or hfov <= 0:
        return 0.0
    dx = egoYaw * imgW / hfov
    if abs(dx) < 0.5:
        return 0.0
    tracker.prev_bboxes = [(x1 + dx, y1, x2 + dx, y2)
                           for (x1, y1, x2, y2) in tracker.prev_bboxes]
    for dq in tracker.history.values():
        papu2 = [(cx + dx, cy) for (cx, cy) in dq]
        dq.clear()
        dq.extend(papu2)
    return dx


class VisualYawEstimator:
    def __init__(self, hfovDeg=70.0, workWidth=640,
                 maxFeatures=200, minFeatures=12):
        self.hfov = math.radians(hfovDeg)
        self.workWidth = workWidth
        self.maxFeatures =maxFeatures
        self.minFeatures = minFeatures
        self.prevGray = None

    def update(self, frameBgr):
        import cv2
        h, w = frameBgr.shape[:2]
        escala = self.workWidth / float(w)
        gray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.workWidth, int(h * escala)))

        prev, self.prevGray = self.prevGray, gray
        if prev is None:
            return 0.0, False

        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self.maxFeatures,
                                      qualityLevel=0.01, minDistance=12)
        if pts is None or len(pts) < self.minFeatures:
            return 0.0, False

        nxt, estado, _err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        buenos = estado.ravel() == 1
        if buenos.sum() < self.minFeatures:
            return 0.0, False

        dx = (nxt[buenos, 0, 0] - pts[buenos, 0, 0])
        fPx = (self.workWidth / 2.0) / math.tan(self.hfov / 2.0)
        deltaYaw= math.atan(float(np.median(dx)) / fPx)
        return deltaYaw, True


class GroundHazardDetector:
    MIN_AREA_FRAC = 0.0015
    MAX_AREA_FRAC = 0.20
    MIN_SURROUND = 0.55

    MIN_DOOR_FRAC = 0.004      # una puerta decente ocupa algo de imagen
    DOOR_OPEN_FLOOR = 0.07     # % de piso DENTRO del bbox = se ve a traves

    def __init__(self, hfovDeg=None, vfovDeg=None,
                 cameraProfile="laptop",
                 camHeightM=DEFAULT_CAM_HEIGHT_M,
                 camPitchDeg=DEFAULT_CAM_PITCH_DEG,
                 workWidth=480, maxRangeM=6.0, detectarPuertas=True):
        # vfovDeg se acepta por compat pero ya no se usa: la geometria
        # sale del hfov + el aspecto real del frame (pixeles cuadrados)
        perfil = CAMERA_PROFILES.get(cameraProfile,
                                     CAMERA_PROFILES["laptop"])
        self.hfov = math.radians(hfovDeg if hfovDeg is not None
                                 else perfil["hfov_deg"])
        self.camHeightM = camHeightM
        self.camPitchDeg = camPitchDeg
        self.workWidth = workWidth
        self.maxRangeM = maxRangeM
        self.detectarPuertas = detectarPuertas
        self._segmenter = None

    def _cargarModelo(self):
        if self._segmenter is None:
            from vision.scene_segmentation import SceneSegmenter
            self._segmenter = SceneSegmenter()

    def process(self, frameBgr, yoloDetections=None):
        import cv2
        self._cargarModelo()

        hOrig, wOrig = frameBgr.shape[:2]
        esc = self.workWidth / float(wOrig)
        chiquito = cv2.resize(frameBgr,
                              (self.workWidth, int(round(hOrig * esc))),
                              interpolation=cv2.INTER_AREA)
        h, w = chiquito.shape[:2]
        self._fPx = 0.5 * w / math.tan(self.hfov / 2.0)
        self._vfovEff = 2.0 * math.atan(0.5 * h / self._fPx)
        seg = self._segmenter.segment(chiquito)
        walkable, water, stairs = self._segmenter.extract_ground_masks(seg)

        peligros = []
        peligros += self._peligrosMascara(water, "agua", w, h)
        peligros += self._peligrosMascara(stairs, "escalera", w, h)
        peligros += self._hoyos(walkable, water, stairs, w, h)

        if yoloDetections:
            peligros = [hz for hz in peligros
                        if not self._chocaConDet(hz, yoloDetections)]

        if self.detectarPuertas:
            peligros += self._puertas(seg, walkable, w, h,
                                      yoloDetections or [])
        return peligros

    def _puertas(self, seg, walkable, w, h, yaDetectadas):
        """salidas: la clase door de ade20k. abierta si dentro del bbox
        se alcanza a ver piso (o sea, se ve a traves del vano)."""
        import cv2
        doorMask = np.isin(seg, self._segmenter.DOOR_IDS).astype(np.uint8)
        n, labelsImg = cv2.connectedComponents(doorMask)
        out = []
        area = w * h
        for i in range(1, n):
            comp = labelsImg == i
            if comp.sum() < self.MIN_DOOR_FRAC * area:
                continue
            ys, xs = np.nonzero(comp)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            caja = walkable[y1:y2 + 1, x1:x2 + 1]
            abierta = float(caja.mean()) > self.DOOR_OPEN_FLOOR
            det = self._compADet(comp, "door", w, h, hazard=False)
            if det is None:
                continue
            det.estado = "abierta" if abierta else "cerrada"
            det.confidence = 0.6
            # si el yolo extra ya la vio, nos quedamos con una sola
            if any(d.label == "door"
                   and abs(d.bearing - det.bearing) < math.radians(8)
                   for d in yaDetectadas):
                continue
            out.append(det)
        return out

    def _compADet(self, compMask, label, w, h, hazard=True):
        ys, xs = np.nonzero(compMask)
        if len(xs) == 0:
            return None
        cx = float(xs.mean())
        filaAbajo = float(ys.max())
        dist = groundDistanceFromRow(
            filaAbajo, h, self._vfovEff, self.camHeightM, self.camPitchDeg)
        if dist is None or dist > self.maxRangeM:
            return None
        bearing = ((w / 2.0 - cx) / w) * self.hfov
        return VisionDetection(
            bearing=bearing, label=label,
            pixel_width=int(xs.max() - xs.min()),
            range_est=max(0.2, dist), moving=False,
            confidence=0.6, hazard=hazard)

    def _peligrosMascara(self, mask, label, w, h):
        import cv2
        mask = mask.copy()
        mask[: int(h * 0.45), :] = 0
        n, labelsImg = cv2.connectedComponents(mask.astype(np.uint8))
        out = []
        area = w * h
        for i in range(1, n):
            comp = labelsImg == i
            frac = comp.sum() / area
            if frac < self.MIN_AREA_FRAC:
                continue
            det = self._compADet(comp, label, w, h)
            if det is not None:
                out.append(det)
        return out

    def _hoyos(self, walkable, water, stairs, w, h):
        import cv2
        area = w * h
        banda = np.zeros_like(walkable)
        banda[int(h * 0.45):, :] = 1
        noPiso = ((walkable == 0) & (water == 0) & (stairs == 0)
                  & (banda == 1)).astype(np.uint8)

        n, labelsImg = cv2.connectedComponents(noPiso)
        out = []
        kernel = np.ones((5, 5), np.uint8)
        for i in range(1, n):
            comp = (labelsImg == i)
            frac = comp.sum() / area
            if frac < self.MIN_AREA_FRAC or frac > self.MAX_AREA_FRAC:
                continue
            ys, xs = np.nonzero(comp)
            if xs.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
                continue
            compU8 = comp.astype(np.uint8)
            anillo = cv2.dilate(compU8, kernel) - compU8
            anilloPx = anillo.sum()
            if anilloPx == 0:
                continue
            alrededor = float((walkable[anillo == 1]).sum()) / anilloPx
            if alrededor < self.MIN_SURROUND:
                continue
            det = self._compADet(comp, "hoyo", w, h)
            if det is not None:
                out.append(det)
        return out

    def _chocaConDet(self, hazard, detections,
                     tol=math.radians(6)):
        for det in detections:
            if det.hazard:
                continue
            if abs(det.bearing - hazard.bearing) < tol and (
                    det.range_est is None or hazard.range_est is None
                    or abs(det.range_est - hazard.range_est)
                    < max(0.8, 0.4 * det.range_est)):
                return True
        return False


HIGH_RISK_LABELS = {"escalera", "hoyo", "agua", "stairs"}


def detectionRisk(det: VisionDetection):
    # tabla 7.1 del TT: movil/escalera/desnivel=alto, estatico=medio,
    # puerta cerrada=medio, puerta abierta=bajo, pared=bajo
    if det.hazard or det.label in HIGH_RISK_LABELS:
        return 3
    if det.moving:
        return 3
    if det.label == "door" and det.estado == "abierta":
        return 1
    if det.label in ("wall", "eco_us", "libre"):
        return 1
    return 2


# junta vision + ultrasonido en obs pal ekf
def fuseObservations(visionDets, ultrasonicReadings,
                     bearingTol=math.radians(20)):
    observaciones = []
    usValidos = [u for u in ultrasonicReadings if u.valid]
    usUsados = set()

    for det in visionDets:
        if det.moving:
            continue
        if det.hazard:
            if det.range_est is not None:
                observaciones.append(Observation(
                    range_m=det.range_est, bearing=det.bearing,
                    label=det.label,
                    sigma_r=SIGMA_R_GROUND_PLANE_PCT * det.range_est,
                    sigma_phi=SIGMA_PHI_VISION,
                    min_confirm_hits=4, is_hazard=True))
            continue
        mejorI, mejorDiff = None, bearingTol
        for i, u in enumerate(usValidos):
            diff = abs(u.sensor_bearing - det.bearing)
            if diff < mejorDiff and _rangosCompatibles(u.range_m, det.range_est):
                mejorI, mejorDiff = i, diff
        if mejorI is not None:
            u = usValidos[mejorI]
            usUsados.add(mejorI)
            observaciones.append(Observation(
                range_m=u.range_m, bearing=det.bearing, label=det.label,
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_VISION))
        elif det.range_est is not None:
            observaciones.append(Observation(
                range_m=det.range_est, bearing=det.bearing, label=det.label,
                sigma_r=SIGMA_R_MONOCULAR_PCT * det.range_est,
                sigma_phi=SIGMA_PHI_VISION,
                min_confirm_hits=4))

    for i, u in enumerate(usValidos):
        if i not in usUsados:
            observaciones.append(Observation(
                range_m=u.range_m, bearing=u.sensor_bearing, label="eco_us",
                sigma_r=SIGMA_R_ULTRASONIC, sigma_phi=SIGMA_PHI_ULTRASONIC,
                can_init_landmark=False, can_correct=False))

    return observaciones


def _rangosCompatibles(usRange, monoRange):
    if monoRange is None:
        return True
    return abs(usRange - monoRange) <= max(0.6, 0.5 * monoRange)


# rayos sin eco = corredor libre, solo pal mapa
def freeSpaceRays(ultrasonicReadings):
    return [
        Observation(range_m=u.max_range + 0.01, bearing=u.sensor_bearing,
                    label="libre")
        for u in ultrasonicReadings if not u.valid
    ]
