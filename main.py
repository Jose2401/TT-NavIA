"""
main.py — TT 

Detecta y clasifica en tiempo real:
 Obstáculos estáticos y en movimiento (por objeto individual)
 Senalización básica (salidas, emergencia, no pase)
 Barreras / paredes
 Segmentación de escena (ADE20K) como apoyo
 Tamaño y dimensiones relativas por objeto
 Movimiento global + movimiento por objeto
 Clasificación extendida (salida / senal / obstáculo)

"""

import cv2
import time
import numpy as np
from collections import defaultdict

from scene_segmentation import SceneSegmenter
from detector import ObjectDetector
from obstacle_logic import ObstacleClassifier, CATEGORY_COLORS
from motion import GlobalMotionDetector, ObjectMotionTracker
from renderer import draw_mask, draw_detection, draw_hud

# Tracker opcional
#try:
#    from tracker import SimpleTracker
#    USE_ADVANCED_TRACKER = True
#    print("usando tracker")
#except Exception:
#    USE_ADVANCED_TRACKER = False
#    print("usando solo motion")

# Configuración
CAMERA_INDEX = 0
MODEL_PATH = "yolov8s-seg.pt"
CONFIDENCE = 0.40
WINDOW_NAME = "Vision System PRO MAX xd"
SAVE_PATH = "captura.jpg" #por si creamos el datset para el modulo de rutas

# Clases extendidas
EXIT_CLASSES = {"door"}  # nao sirve xd, habria q hacer un data set propio o algo
SIGN_CLASSES = {"stop sign", "traffic light", "no entry", "warning sign"}

# Segmentación
SEGMENT_EVERY_N_FRAMES = 3

# Umbrales
DETECTION_CONF_THRESHOLD = 0.45
BARRIER_OVERRIDE_THRESHOLD = 0.55
PERSON_BARRIER_THRESHOLD = 0.80

# =============================
# Procesamiento de imágenes
# =============================
def preprocesar_imagen(fotograma_crudo, tamano_objetivo=(640, 480)):
    """
    Prepara el fotograma para el módulo de visión:
    - Redimensiona
    - Normaliza a [0,1]
    - Igualación de histograma
    - Filtro gaussiano
    """
    if fotograma_crudo is None or fotograma_crudo.size == 0:
        print("[ERROR] Fotograma vacío en preprocesamiento.")
        return None

    # Redimensionar
    imagen_redimensionada = cv2.resize(fotograma_crudo, tamano_objetivo, interpolation=cv2.INTER_AREA)

    # Normalizar a [0,1]
    imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0

    # Igualación de histograma por canal (en color)
    imagen_ecualizada = np.zeros_like(imagen_normalizada)
    for c in range(3):
        canal = (imagen_normalizada[:, :, c] * 255).astype(np.uint8)
        canal_eq = cv2.equalizeHist(canal)
        imagen_ecualizada[:, :, c] = canal_eq.astype(np.float32) / 255.0

    # Filtro gaussiano
    imagen_lista = cv2.GaussianBlur(imagen_ecualizada, (5, 5), sigmaX=0.5, sigmaY=0.5)

    return imagen_lista

# Es para ver la segmentacion de piso y barrera que hace A20k en otra parte extr a la UI
def build_segmentation_preview(
    barrier_mask: np.ndarray | None,
    floor_mask: np.ndarray | None,
    size: tuple[int, int] = (260, 160),
) -> np.ndarray:
    preview = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    if barrier_mask is None or floor_mask is None:
        cv2.putText(
            preview,
            "Segmentacion",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return preview

    barrier_small = cv2.resize(
        (barrier_mask * 255).astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    )
    floor_small = cv2.resize(
        (floor_mask * 255).astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    )

    preview[:] = (25, 25, 25) #fondo  gris
    preview[floor_small > 0] = (0, 180, 0)
    preview[barrier_small > 0] = (0, 0, 220)

    cv2.rectangle(preview, (0, 0), (size[0] - 1, size[1] - 1), (255, 255, 255), 1)
    cv2.putText(
        preview,
        "ADE20K",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Rojo=barrera | Verde=piso",
        (8, size[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return preview

detector = ObjectDetector(model_path=MODEL_PATH, confidence=CONFIDENCE)
classifier = ObstacleClassifier()
global_motion = GlobalMotionDetector()
obj_tracker = ObjectMotionTracker()

#if USE_ADVANCED_TRACKER:
#    adv_tracker = SimpleTracker()

segmenter = SceneSegmenter()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=" * 60)
print("Sistema PRO con segmentación iniciado")
print("ESC / Q  → salir")
print("S        → guardar captura")
print("=" * 60)

fps_time = time.time()
frame_count = 0

# Cache de segmentación
seg_map = None
barrier_mask = None
floor_mask = None


# Main xd con monitoreo de tiempos
last_result_time = time.time()
while True:
    loop_start = time.time()


    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer frame. Verifica la cámara.")
        break

    # Copia del frame original para debug
    frame_original_debug = frame.copy()

    # Procesamiento de imagen
    frame_proc = preprocesar_imagen(frame, tamano_objetivo=(640, 480))
    if frame_proc is None:
        continue

    frame_h, frame_w = frame_proc.shape[:2]
    frame_count += 1

    # Segmentación de entorno
    t0 = time.time()
    seg_updated = False
    if frame_count % SEGMENT_EVERY_N_FRAMES == 0:
        seg_map = segmenter.segment((frame_proc * 255).astype(np.uint8))
        barrier_mask, floor_mask = segmenter.extract_navigation_mask(seg_map)
        seg_updated = True
    t1 = time.time()

    # Detección (CNN YOLOv8-seg)
    t2 = time.time()
    detections = detector.detect((frame_proc * 255).astype(np.uint8))
    t3 = time.time()

    # Filtro de estabilidad
    detections = [d for d in detections if d["confidence"] > DETECTION_CONF_THRESHOLD]

    # Movimiento global
    t4 = time.time()
    scene_moving, motion_ratio = global_motion.detect((frame_proc * 255).astype(np.uint8))
    t5 = time.time()

    # Movimiento por objeto
    t6 = time.time()
    obj_moving_flags = obj_tracker.update(detections)
    t7 = time.time()

    #if USE_ADVANCED_TRACKER:
    #    adv_flags = adv_tracker.update(detections)
    #else:
    adv_flags = obj_moving_flags

    # Clasificación y render
    t8 = time.time()
    category_counts = defaultdict(int)

    for i, (det, is_obj_moving) in enumerate(zip(detections, obj_moving_flags)):
        #if USE_ADVANCED_TRACKER:
        #    is_obj_moving = adv_flags[i]

        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        # Clase base 
        class_name = det.get("label", det.get("class", "unknown")).lower()
        category = classifier.classify(det, frame_w, frame_h)

        # intento de dimensiones y distancia
        dims = classifier.estimate_dimensions(det["bbox"], frame_w, frame_h)
        dist = classifier.estimate_distance(det["bbox"], frame_h)

        # Tipo semántico
        tipo = "obstaculo"
        if class_name in EXIT_CLASSES:
            tipo = "salida"
        elif class_name in SIGN_CLASSES:
            tipo = "senal"

        # Fusión con segmentación ADE20K
        barrier_ratio = 0.0
        if barrier_mask is not None and x2 > x1 and y2 > y1:
            region = barrier_mask[y1:y2, x1:x2]
            if region.size > 0:
                barrier_ratio = float(region.mean())

        # Si la zona está realmente dominada por barrera, fuerza barrera. Esto falla aveces y marca cosas que no son
        if (
            barrier_ratio >= BARRIER_OVERRIDE_THRESHOLD
            and not (class_name == "person" and barrier_ratio < PERSON_BARRIER_THRESHOLD) # Fix pq las personas tan muy grandes xd
            and tipo == "obstaculo"
        ):
            category = "barrera"

        # Labels de estado y extras a COCO 
        extra_labels = []

        if is_obj_moving:
            extra_labels.append("MOVIMIENTO")

        if tipo == "salida":
            extra_labels.append("SALIDA")
        elif tipo == "senal":
            extra_labels.append("SENAL")

        if scene_moving:
            extra_labels.append("ESCENA DINAMICA")

        if barrier_ratio > 0.0:
            extra_labels.append(f"BARRERA {int(barrier_ratio * 100)}%")

        # ── Render máscara YOLO ────────────────────────────
        frame = draw_mask(frame, det.get("mask"), CATEGORY_COLORS.get(category), alpha=0.18)

        # ── Render bbox ────────────────────────────────────
        frame = draw_detection(
            frame,
            det,
            category,
            is_obj_moving,
            dims + f" | {dist}",
            extra_labels=extra_labels
        )

        category_counts[category] += 1
    t9 = time.time()

    #Mini vista de segmentación (llamada xd)
    if barrier_mask is not None and floor_mask is not None:
        preview = build_segmentation_preview(barrier_mask, floor_mask, size=(240, 140))
        ph, pw = preview.shape[:2]

        x0 = max(0, frame.shape[1] - pw - 12)
        y0 = max(0, frame.shape[0] - ph - 12)

        roi = frame[y0:y0 + ph, x0:x0 + pw]
        if roi.shape[:2] == preview.shape[:2]:
            frame[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(roi, 0.20, preview, 0.80, 0)

    # || HUD ||
    now = time.time()
    fps = 1.0 / max(now - fps_time, 1e-6)
    fps_time = now
    frame = draw_hud(
        frame,
        len(detections),
        motion_ratio,
        fps,
        dict(category_counts)
    )

    # Monitoreo de tiempos y generación de resultados
    render_time = time.time()
    seg_time = (t1 - t0) if seg_updated else 0.0
    det_time = t3 - t2
    global_motion_time = t5 - t4
    obj_motion_time = t7 - t6
    render_stage_time = t9 - t8
    total_loop_time = render_time - loop_start
    time_since_last_result = render_time - last_result_time
    last_result_time = render_time

    print(f"[MONITOREO] Frame {frame_count} | Segmentación: {seg_time:.3f}s | Detección: {det_time:.3f}s | GlobalMotion: {global_motion_time:.3f}s | ObjMotion: {obj_motion_time:.3f}s | Render: {render_stage_time:.3f}s | Total: {total_loop_time:.3f}s | Δt desde último resultado: {time_since_last_result:.3f}s \n")

    cv2.imshow(WINDOW_NAME, frame)


    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):
        break
    elif key == ord("s"):
        cv2.imwrite(SAVE_PATH, frame)
        print(f"[INFO] Captura guardada en {SAVE_PATH}")
        print(f"[INFO] Objetos detectados: {len(detections)}")
    elif key == ord("d"):
        # Guardar frame original y preprocesado
        nombre_base = f"img_filtros/debug_frame_{frame_count}"
        cv2.imwrite(f"{nombre_base}_original.jpg", frame_original_debug)
        # El preprocesado está en float32 [0,1], convertir a uint8 para guardar
        frame_proc_uint8 = (np.clip(frame_proc, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"{nombre_base}_preprocesado.jpg", frame_proc_uint8)
        print(f"[DEBUG] Guardadas capturas: {nombre_base}_original.jpg y {nombre_base}_preprocesado.jpg")

# ── Limpieza ────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Sistema detenido.")