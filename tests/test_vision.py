"""
pruebas del modulo de vision mejorado: geometria sin distorsion de
aspecto, detector extra (oiv7) con merge, puertas por segmentacion con
estado abierta/cerrada, y riesgos. sin modelos reales (fakes).
correr: python -m unittest tests.test_vision
"""
import math
import unittest

import numpy as np

from interfaces import VisionDetection
from vision_bridge import (VisionBridge, GroundHazardDetector,
                           detectionRisk, groundDistanceFromRow,
                           EXTRA_OIV7, _iou, TYPICAL_HEIGHTS_M)


class FakeYolo:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame):
        return self.dets


def _bridgeSinModelos(dets, hfovDeg=70.0, extras=None):
    """arma un VisionBridge sin cargar yolo de verdad."""
    b = VisionBridge.__new__(VisionBridge)
    b.detector = FakeYolo(dets)
    b.aliases = {}
    b.hfov = math.radians(hfovDeg)
    b.vfov = math.radians(55)
    b.inferWidth = 640
    b.camHeightM = 1.55
    b.camPitchDeg = 10.0
    b.useGroundPlane = False
    b.extraDetector = None
    b.extraEvery = 1
    b._nFrame = 0
    b._extraIds = {}
    from vision.motion import ObjectMotionTracker
    b.tracker = ObjectMotionTracker()
    if extras is not None:
        class FakeExtra:
            confidence = 0.3
        b.extraDetector = FakeExtra()
        b._detectarExtras = lambda frame: extras
    return b


def _frame169():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


class TestGeometria(unittest.TestCase):

    def test_resize_conserva_aspecto(self):
        b = _bridgeSinModelos([])
        chico = b._redimensionar(_frame169())
        h, w = chico.shape[:2]
        self.assertEqual(w, 640)
        self.assertEqual(h, 360)      # 16:9, no 480 estirado

    def test_distancia_por_altura_correcta_en_16a9(self):
        """persona de 1.65 m a 3 m con hfov 70: la bbox debe dar ~3 m,
        no ~2.2 como salia con el frame estirado a 4:3."""
        hfov = math.radians(70)
        fPx = 0.5 * 640 / math.tan(hfov / 2)      # focal en px (w=640)
        dist = 3.0
        hPx = fPx * 1.65 / dist                   # alto esperado en px
        det = {"label": "person", "confidence": 0.9,
               "bbox": (300, 100, 340, int(100 + hPx)), "mask": None}
        b = _bridgeSinModelos([det])
        out = b.process(_frame169())
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].range_est, 3.0, delta=0.25)

    def test_bearing_centro_e_izquierda(self):
        det = {"label": "chair", "confidence": 0.9,
               "bbox": (310, 100, 330, 300), "mask": None}   # centrado
        b = _bridgeSinModelos([det])
        out = b.process(_frame169())
        self.assertAlmostEqual(out[0].bearing, 0.0, delta=math.radians(2))
        det2 = {"label": "chair", "confidence": 0.9,
                "bbox": (0, 100, 40, 300), "mask": None}     # borde izq
        b2 = _bridgeSinModelos([det2])
        out2 = b2.process(_frame169())
        self.assertGreater(out2[0].bearing, math.radians(25))


class TestDetectorExtra(unittest.TestCase):

    def test_extra_agrega_puerta(self):
        extra = [{"label": "door", "confidence": 0.5,
                  "bbox": (100, 50, 200, 350), "mask": None}]
        b = _bridgeSinModelos([], extras=extra)
        out = b.process(_frame169())
        self.assertEqual([d.label for d in out], ["door"])
        self.assertFalse(out[0].hazard)

    def test_extra_escalera_es_hazard(self):
        extra = [{"label": "escalera", "confidence": 0.5,
                  "bbox": (100, 200, 300, 350), "mask": None}]
        b = _bridgeSinModelos([], extras=extra)
        out = b.process(_frame169())
        self.assertTrue(out[0].hazard)
        self.assertEqual(detectionRisk(out[0]), 3)

    def test_extra_no_duplica_lo_de_coco(self):
        coco = {"label": "chair", "confidence": 0.9,
                "bbox": (100, 100, 200, 300), "mask": None}
        extra = [{"label": "shelf", "confidence": 0.5,
                  "bbox": (105, 105, 205, 305), "mask": None}]  # mismo lugar
        b = _bridgeSinModelos([coco], extras=extra)
        out = b.process(_frame169())
        self.assertEqual([d.label for d in out], ["chair"])

    def test_extra_respeta_cadencia(self):
        llamadas = []
        extra = [{"label": "door", "confidence": 0.5,
                  "bbox": (100, 50, 200, 350), "mask": None}]
        b = _bridgeSinModelos([], extras=extra)
        b.extraEvery = 3
        real = b._detectarExtras
        b._detectarExtras = lambda f: (llamadas.append(1), real(f))[1]
        for _ in range(6):
            b.process(_frame169())
        self.assertEqual(len(llamadas), 2)

    def test_mapa_de_clases_oiv7(self):
        self.assertEqual(EXTRA_OIV7["Door"], "door")
        self.assertEqual(EXTRA_OIV7["Stairs"], "escalera")
        for etiqueta in EXTRA_OIV7.values():
            # toda clase extra debe tener altura tipica o ser hazard
            self.assertTrue(etiqueta in TYPICAL_HEIGHTS_M
                            or etiqueta == "escalera", etiqueta)


class FakeSegmenterPuertas:
    DOOR_IDS = [14]

    def __init__(self, abierta):
        self.abierta = abierta

    def segment(self, frame):
        h, w = frame.shape[:2]
        seg = np.zeros((h, w), dtype=np.int64)     # 0 = wall
        seg[int(h*0.7):, :] = 3                    # piso abajo
        # puerta al centro: marco (door) con interior segun el caso
        x1, x2 = int(w*0.40), int(w*0.60)
        y1, y2 = int(h*0.15), int(h*0.80)
        seg[y1:y2, x1:x2] = 14
        if self.abierta:
            # se ve piso a traves del vano
            seg[int(h*0.45):y2, int(w*0.44):int(w*0.56)] = 3
        return seg

    def extract_ground_masks(self, seg):
        walk = np.isin(seg, [3]).astype(np.uint8)
        nada = np.zeros_like(walk)
        return walk, nada, nada


class TestPuertasPorSegmentacion(unittest.TestCase):

    def _detector(self, abierta):
        d = GroundHazardDetector(cameraProfile="laptop")
        d._segmenter = FakeSegmenterPuertas(abierta)
        return d

    def test_puerta_cerrada(self):
        dets = self._detector(False).process(np.zeros((360, 480, 3), np.uint8))
        puertas = [d for d in dets if d.label == "door"]
        self.assertEqual(len(puertas), 1)
        self.assertEqual(puertas[0].estado, "cerrada")
        self.assertFalse(puertas[0].hazard)
        self.assertEqual(detectionRisk(puertas[0]), 2)

    def test_puerta_abierta_es_riesgo_bajo(self):
        dets = self._detector(True).process(np.zeros((360, 480, 3), np.uint8))
        puertas = [d for d in dets if d.label == "door"]
        self.assertEqual(len(puertas), 1)
        self.assertEqual(puertas[0].estado, "abierta")
        self.assertEqual(detectionRisk(puertas[0]), 1)

    def test_no_duplica_si_yolo_ya_la_vio(self):
        det = self._detector(False)
        ya = [VisionDetection(bearing=0.0, label="door", range_est=3.0)]
        dets = det.process(np.zeros((360, 480, 3), np.uint8),
                           yoloDetections=ya)
        self.assertEqual([d for d in dets if d.label == "door"], [])

    def test_puertas_desactivables(self):
        d = self._detector(False)
        d.detectarPuertas = False
        dets = d.process(np.zeros((360, 480, 3), np.uint8))
        self.assertEqual([x for x in dets if x.label == "door"], [])


class TestVarios(unittest.TestCase):

    def test_iou(self):
        self.assertAlmostEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_ground_distance_sigue_igual(self):
        vfov = math.radians(60)
        f = 0.5 * 480 / math.tan(vfov / 2)
        row = 400
        esperado = 1.5 / math.tan(math.atan((row - 240) / f))
        self.assertAlmostEqual(
            groundDistanceFromRow(row, 480, vfov, 1.5, 0.0),
            esperado, places=6)

    def test_alias_de_salidas_en_navigator(self):
        from navigation.navigator import DEST_ALIASES
        self.assertEqual(DEST_ALIASES["salida"], "door")
        self.assertEqual(DEST_ALIASES["ventana"], "window")
        self.assertEqual(DEST_ALIASES["escaleras"], "escalera")
        self.assertEqual(DEST_ALIASES["basura"], "trash can")


if __name__ == "__main__":
    unittest.main()


class TestMuebleNoSeMueve(unittest.TestCase):
    """2026-11-12, gafas quietas: una cama salia 'en movimiento' y el
    sistema decia 'Alto. Se acerca algo al frente'."""

    def _dets(self, dx):
        return [{"label": "bed", "confidence": 0.8,
                 "bbox": (100 + dx, 150, 400 + dx, 330), "mask": None},
                {"label": "person", "confidence": 0.8,
                 "bbox": (450, 80, 520, 330), "mask": None}]

    def test_solo_lo_vivo_va_en_movimiento(self):
        b = _bridgeSinModelos(self._dets(0))
        out = []
        for dx in (0, 12, 24, 36):          # la caja de la cama se corre
            b.detector = FakeYolo(self._dets(dx))
            out = b.process(_frame169())
        cama = [d for d in out if d.label == "bed"][0]
        persona = [d for d in out if d.label == "person"][0]
        self.assertFalse(cama.moving)
        self.assertTrue(persona.moving)

    def test_se_puede_volver_a_prender_el_rastreo(self):
        b = _bridgeSinModelos(self._dets(0))
        b.moverEstaticos = True
        out = []
        for dx in (0, 12, 24, 36):
            b.detector = FakeYolo(self._dets(dx))
            out = b.process(_frame169())
        self.assertTrue([d for d in out if d.label == "bed"][0].moving)


class TestPuertasYClasesFalsas(unittest.TestCase):
    """recorridos reales del 2026-11-12 (oficina y salon)."""

    def test_la_puerta_del_extra_corrige_al_refrigerador(self):
        """la puerta amarilla del salon salia como 'refrigerador' (0.94)
        y la puerta que SI veia el detector extra se tiraba por encimarse."""
        caja = (220, 110, 330, 300)
        coco = [{"label": "refrigerator", "confidence": 0.94,
                 "bbox": caja, "mask": None}]
        puerta = [{"label": "door", "confidence": 0.41,
                   "bbox": (225, 112, 328, 298), "mask": None}]
        b = _bridgeSinModelos(coco, extras=puerta)
        b.aliases = {"refrigerator": "fridge"}
        b.extraEvery = 3
        etiquetas = []
        for _ in range(6):               # el extra corre en el 3 y el 6
            b.detector = FakeYolo([dict(coco[0])])
            etiquetas.append([d.label for d in b.process(_frame169())])
        self.assertEqual(etiquetas[0], ["fridge"])       # aun no corre
        for e in etiquetas[2:]:                          # y se recuerda
            self.assertEqual(e, ["door"])

    def test_a_una_persona_no_se_le_cambia_el_nombre(self):
        caja = (220, 110, 330, 300)
        coco = [{"label": "person", "confidence": 0.9, "bbox": caja,
                 "mask": None}]
        puerta = [{"label": "door", "confidence": 0.41, "bbox": caja,
                   "mask": None}]
        b = _bridgeSinModelos(coco, extras=puerta)
        self.assertEqual([d.label for d in b.process(_frame169())],
                         ["person"])

    def test_clases_imposibles_no_pasan(self):
        dets = [{"label": "parking meter", "confidence": 0.91,
                 "bbox": (500, 10, 600, 340), "mask": None},
                {"label": "train", "confidence": 0.85,
                 "bbox": (10, 100, 200, 300), "mask": None},
                {"label": "chair", "confidence": 0.8,
                 "bbox": (300, 200, 360, 320), "mask": None}]
        b = _bridgeSinModelos(dets)
        b.letrerosSalida = False
        self.assertEqual([d.label for d in b.process(_frame169())],
                         ["chair"])

    def test_lo_que_se_carga_no_es_marca_del_mapa(self):
        from interfaces import VisionDetection
        from vision_bridge import fuseObservations
        dets = [VisionDetection(0.0, "backpack", range_est=2.0),
                VisionDetection(0.3, "chair", range_est=2.0)]
        self.assertEqual([o.label for o in fuseObservations(dets, [])],
                         ["chair"])

    def test_letrero_verde_de_salida(self):
        import cv2
        from vision_bridge import detectarLetrerosSalida
        img = np.full((360, 640, 3), 200, dtype=np.uint8)
        cv2.rectangle(img, (400, 100), (470, 150), (60, 160, 0), -1)   # verde
        cv2.arrowedLine(img, (455, 125), (415, 125), (255, 255, 255), 6)
        cv2.circle(img, (100, 250), 60, (40, 140, 30), -1)   # un arbol no
        r = detectarLetrerosSalida(img)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["label"], "exit sign")
        x1, y1, x2, y2 = r[0]["bbox"]
        self.assertTrue(395 <= x1 <= 405 and 465 <= x2 <= 475)
