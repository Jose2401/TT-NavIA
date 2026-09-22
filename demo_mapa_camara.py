"""
demo_mapa_camara.py
Mapeo en TIEMPO REAL usando SOLO LA CÁMARA: sin sensores ultrasónicos,
sin IMU, sin odometría. Es la prueba del pipeline completo con el único
hardware disponible hoy:

 1. La cámara alimenta al módulo de visión REAL (vision/TT-NavIA,
    YOLOv8-seg vía vision_bridge.VisionBridge): objetos detectados, su
    dirección (bearing) y su DISTANCIA MONOCULAR estimada por el tamaño
    del bbox (con incertidumbre grande, que el EKF pondera).
 2. El GIRO del usuario ("hacia dónde voltea") se estima de la propia
    imagen por flujo óptico (vision_bridge.VisualYawEstimator): al girar
    la cabeza, el mapa y theta se actualizan aunque no haya IMU.
 3. El EKF integra giro + observaciones monoculares; la posición (x, y)
    se ajusta con las distancias a los objetos ya vistos (si te acercas a
    la silla, el rango baja y la pose avanza). Sin odometría la posición
    es gruesa — mejorará sola cuando exista ese módulo (TODO), sin tocar
    este script salvo sustituir el MotionEstimate.
 4. El mapa de ocupación se construye en vivo (panel derecho) y al salir
    se genera el croquis con la trayectoria y los landmarks.

Uso:
    python demo_mapa_camara.py                 # cámara (índice 0)
    python demo_mapa_camara.py --camera 1
    python demo_mapa_camara.py --video paseo.mp4   # o un video grabado
    python demo_mapa_camara.py --video paseo.mp4 --max-frames 200 --headless

Controles: q -> salir y generar croquis_camara.png
"""
import argparse
import math
import time

import numpy as np
import cv2

from ekf_slam import EKFSlam
from occupancy_grid import OccupancyGrid
from interfaces import MotionEstimate
from vision_bridge import VisionBridge, VisualYawEstimator, fuse_observations


def draw_map_panel(grid, slam, output, size=480):
    """Panel del mapa de ocupación con pose estimada (rojo), elipse de
    incertidumbre 2-sigma y landmarks etiquetados (naranja)."""
    occ_prob = 1.0 / (1.0 + np.exp(-grid.log_odds))
    img = (255 * (1 - occ_prob)).astype(np.uint8)
    img = np.flipud(img)                        # "arriba" = +y
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = size / float(grid.n)
    view = cv2.resize(view, (size, size), interpolation=cv2.INTER_NEAREST)

    def to_px(wx, wy):
        gx, gy = grid.world_to_grid(wx, wy)
        return int(gx * s), int((grid.n - 1 - gy) * s)

    for lm in slam.get_landmarks():
        px, py = to_px(lm["x"], lm["y"])
        cv2.drawMarker(view, (px, py), (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(view, lm["label"], (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1,
                    cv2.LINE_AA)

    # Elipse de incertidumbre 2-sigma de la posición
    P = np.array(output.covariance)[:2, :2]
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, 0.0)
    axes = (max(2, int(2 * math.sqrt(vals[1]) / grid.resolution * s)),
            max(2, int(2 * math.sqrt(vals[0]) / grid.resolution * s)))
    angle = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    px, py = to_px(output.x, output.y)
    cv2.ellipse(view, (px, py), axes, angle, 0, 360, (0, 0, 200), 1,
                cv2.LINE_AA)

    # Pose estimada + flecha de orientación ("hacia dónde voltea")
    cv2.circle(view, (px, py), 6, (0, 0, 255), -1)
    cv2.line(view, (px, py),
             (int(px + 24 * math.cos(output.theta)),
              int(py - 24 * math.sin(output.theta))), (0, 0, 255), 2)

    cv2.putText(view,
                f"x={output.x:+.2f}m y={output.y:+.2f}m "
                f"theta={math.degrees(output.theta):+.0f}deg",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60), 1,
                cv2.LINE_AA)
    cv2.putText(view,
                f"conf={output.confidence:.2f}  "
                f"landmarks={len(slam.confirmed_landmarks())}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60), 1,
                cv2.LINE_AA)
    return view


def annotate_camera(frame, detections, hfov, yaw_ok, fps):
    """Marca cada detección en su columna (verde=estática usable,
    rojo=en movimiento, descartada como landmark) con etiqueta y
    distancia monocular estimada."""
    h, w = frame.shape[:2]
    for det in detections:
        col = int(w / 2.0 - (det.bearing / hfov) * w)
        color = (0, 0, 255) if det.moving else (0, 255, 0)
        cv2.line(frame, (col, int(h * 0.30)), (col, int(h * 0.70)), color, 2)
        tag = f"{det.label} {det.range_est:.1f}m"
        if det.moving:
            tag += " [movil]"
        cv2.putText(frame, tag, (max(0, col - 50), int(h * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    status = "giro visual: OK" if yaw_ok else "giro visual: SIN TEXTURA"
    cv2.putText(frame, f"{status} | {fps:.1f} fps | q: croquis y salir",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0,
                        help="índice de la cámara")
    parser.add_argument("--video", default=None,
                        help="usar un archivo de video en vez de la cámara")
    parser.add_argument("--model", default=None,
                        help="pesos YOLOv8-seg (por defecto los de TT-NavIA)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="terminar tras N frames (pruebas)")
    parser.add_argument("--headless", action="store_true",
                        help="sin ventana (pruebas con --video)")
    args = parser.parse_args()

    print("Cargando módulo de visión (YOLOv8-seg de vision/TT-NavIA)...")
    bridge = VisionBridge(model_path=args.model)
    yaw_est = VisualYawEstimator(hfov_deg=math.degrees(bridge.hfov))

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {source}")

    # El usuario arranca en el origen del mapa mirando hacia +x.
    slam = EKFSlam(0.0, 0.0, 0.0)
    grid = OccupancyGrid(size_m=10.0, resolution=0.05)
    trajectory = []
    output = slam.get_output()
    last_t = time.time()
    fps = 0.0
    frames = 0

    print("Mapeando SOLO con la cámara (sin ultrasonido/IMU). "
          "Gira lentamente para barrer el cuarto. 'q' para terminar.")
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break               # fin del archivo
                print("[aviso] Frame de cámara perdido.")
                continue
            frames += 1

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # 1) Giro estimado visualmente (flujo óptico). Traslación: 0
            #    hasta que exista odometría/IMU (TODO); el EKF ajusta
            #    (x, y) con las distancias a los objetos conocidos.
            delta_rot, yaw_ok = yaw_est.update(frame)
            motion = MotionEstimate(delta_trans=0.0, delta_rot=delta_rot,
                                    dt=dt)

            # 2) Visión real: objetos + bearing + distancia monocular
            detections = bridge.process(frame)
            observations = fuse_observations(detections, [])  # sin US

            # 3) Ciclo SLAM + mapa. Al mapa van las observaciones ANCLADAS
            #    a los landmarks consolidados: el mismo objeto se pinta
            #    siempre en la misma celda (sin duplicados ni manchas).
            output = slam.step(motion, observations)
            grid.update_from_scan((output.x, output.y, output.theta),
                                  slam.observations_for_map())
            trajectory.append((output.x, output.y))

            # 4) Presentación
            if not args.headless:
                frame_v = annotate_camera(frame, detections, bridge.hfov,
                                          yaw_ok, fps)
                map_v = draw_map_panel(grid, slam, output,
                                       size=frame_v.shape[0])
                cv2.imshow("SLAM solo-camara: vision | mapa",
                           np.hstack([frame_v, map_v]))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
            elif frames % 30 == 0:
                print(f"[f={frames:04d}] theta="
                      f"{math.degrees(output.theta):+6.1f}deg "
                      f"conf={output.confidence:.2f} "
                      f"obs={len(observations)} "
                      f"landmarks={len(slam.confirmed_landmarks())}")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    path = grid.render_croquis(
        pose=(output.x, output.y, output.theta),
        trajectory=trajectory,
        landmarks=slam.confirmed_landmarks(),
        path="croquis_camara.png")
    print(f"\nCroquis guardado en: {path}")
    print(f"Pose final estimada: x={output.x:.2f} m, y={output.y:.2f} m, "
          f"theta={math.degrees(output.theta):.1f} grados "
          f"(confianza={output.confidence:.2f})")
    print(f"Landmarks del recorrido: {slam.get_landmarks()}")


if __name__ == "__main__":
    main()
