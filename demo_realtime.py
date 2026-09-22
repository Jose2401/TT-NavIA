"""
demo_realtime.py
Interfaz en tiempo real del SLAM: pose (x, y, theta) + confianza + mapa de
ocupación, en dos modos:

    python demo_realtime.py            # cámara real + módulo de visión real
                                       # (vision/TT-NavIA, YOLOv8-seg)
    python demo_realtime.py --sim      # sin hardware: cuarto simulado
                                       # (sim_world.py), mismo pipeline

Controles:
    q -> termina y genera el "croquis" (PNG) del mapa construido, con la
         trayectoria y la posición/orientación final del usuario.

Estado de integración (ver DISENO_SLAM_EKF.md §17):
    - Visión: REAL (vision_bridge.VisionBridge sobre vision/TT-NavIA).
    - Ultrasonido: aún no hay hardware -> en modo cámara no se usan ecos
      (los rangos vienen de la estimación monocular de visión, con mayor
      incertidumbre). TODO: conectar HC-SR04 y pasar las lecturas a
      fuse_observations(), sin tocar nada más.
    - Movimiento: aún no hay odometría/IMU -> en modo cámara se asume al
      usuario cuasi-estático (MotionEstimate(0,0)); el EKF sigue sumando
      ruido de proceso y corrigiendo con lo que ve. TODO: conectar el
      módulo real de estimación de movimiento.
"""
import argparse
import math
import time

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ekf_slam import EKFSlam
from occupancy_grid import OccupancyGrid
from interfaces import MotionEstimate
from vision_bridge import fuse_observations, free_space_rays
from sim_world import (SimulatedWorld, SimulatedMotion,
                       SimulatedUltrasonicArray, simulated_vision)


def draw_map_view(grid, pose, landmarks=(), true_pose=None, size=420):
    """Vista del mapa de ocupación (BGR, size x size) con la posición y
    orientación estimadas del usuario (rojo), landmarks (naranja) y, en
    simulación, la pose real (verde)."""
    occ_prob = 1.0 / (1.0 + np.exp(-grid.log_odds))
    img = (255 * (1 - occ_prob)).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def to_px(wx, wy):
        gx, gy = grid.world_to_grid(wx, wy)
        return gx, grid.n - 1 - gy          # "arriba" = +y

    for (lx, ly) in landmarks:
        px, py = to_px(lx, ly)
        if 0 <= px < grid.n and 0 <= py < grid.n:
            cv2.drawMarker(img_bgr, (px, py), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 6, 1)

    def draw_pose(p, color):
        x, y, theta = p
        px, py = to_px(x, y)
        cv2.circle(img_bgr, (px, py), 4, color, -1)
        ex = int(px + 12 * math.cos(theta))
        ey = int(py - 12 * math.sin(theta))
        cv2.line(img_bgr, (px, py), (ex, ey), color, 2)

    if true_pose is not None:
        draw_pose(true_pose, (0, 180, 0))
    draw_pose(pose, (0, 0, 255))

    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_NEAREST)


def annotate_frame(frame, detections, hfov, output):
    """Dibuja sobre el frame de cámara una marca vertical por detección
    (en la columna correspondiente a su bearing) y el estado del SLAM."""
    h, w = frame.shape[:2]
    for det in detections:
        col = int(w / 2.0 - (det.bearing / hfov) * w)
        color = (0, 0, 255) if det.moving else (0, 255, 0)
        cv2.line(frame, (col, int(h * 0.25)), (col, int(h * 0.75)), color, 2)
        tag = f"{det.label} {det.range_est:.1f}m" if det.range_est else det.label
        if det.moving:
            tag += " [movil]"
        cv2.putText(frame, tag, (max(0, col - 40), int(h * 0.23)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(frame,
                f"pose=({output.x:+.2f}, {output.y:+.2f}) "
                f"theta={math.degrees(output.theta):+.0f} deg  "
                f"conf={output.confidence:.2f}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def finish(grid, output, trajectory, landmarks, gt_traj=None,
           path="croquis_mapa.png"):
    path = grid.render_croquis(
        pose=(output.x, output.y, output.theta),
        trajectory=trajectory, landmarks=landmarks,
        gt_trajectory=gt_traj, path=path)
    print(f"\nCroquis del mapa guardado en: {path}")
    print(f"Pose final estimada: x={output.x:.2f} m, y={output.y:.2f} m, "
          f"theta={math.degrees(output.theta):.1f}°  "
          f"(confianza={output.confidence:.2f})")


# ----------------------------------------------------------------------
# Modo simulado: cuarto sintético, mismo pipeline que el modo real
# ----------------------------------------------------------------------
def run_sim(max_steps=None):
    world = SimulatedWorld.default_room()
    motion_sim = SimulatedMotion(start=(0.7, 0.7, 0.0))
    us_array = SimulatedUltrasonicArray(world)
    slam = EKFSlam(x0=0.7, y0=0.7, theta0=0.0)
    grid = OccupancyGrid(size_m=8.0, resolution=0.05, origin=(1.0, 1.0))

    trajectory, gt_traj = [], []
    output = slam.get_output()
    step = 0

    print("Modo simulado. Presiona 'q' para salir y generar el croquis.")
    while max_steps is None or step < max_steps:
        step += 1
        motion = motion_sim.get()
        true_pose = tuple(motion_sim.true_pose)
        us_readings = us_array.read(true_pose)
        vision_dets = simulated_vision(world, true_pose)

        observations = fuse_observations(vision_dets, us_readings)
        output = slam.step(motion, observations)
        grid.update_from_scan((output.x, output.y, output.theta),
                              slam.observations_for_map()
                              + free_space_rays(us_readings))
        trajectory.append((output.x, output.y))
        gt_traj.append((true_pose[0], true_pose[1]))

        if HAS_CV2:
            view = draw_map_view(grid, (output.x, output.y, output.theta),
                                 landmarks=slam.confirmed_landmarks(),
                                 true_pose=true_pose)
            cv2.putText(view, f"conf={output.confidence:.2f}  "
                        f"landmarks={len(slam.confirmed_landmarks())}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("SLAM (simulacion) - rojo: estimado, verde: real", view)
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                break
        else:
            time.sleep(motion.dt)
            if step >= (max_steps or 300):
                break

    if HAS_CV2:
        cv2.destroyAllWindows()
    finish(grid, output, trajectory, slam.confirmed_landmarks(),
           gt_traj=gt_traj)


# ----------------------------------------------------------------------
# Modo real: cámara + vision/TT-NavIA (YOLOv8-seg) vía vision_bridge
# ----------------------------------------------------------------------
def run_camera(camera_index=0, model_path=None):
    if not HAS_CV2:
        raise SystemExit("OpenCV no está disponible; usa --sim.")

    from vision_bridge import VisionBridge   # import diferido (carga YOLO)
    print("Cargando módulo de visión (YOLOv8-seg)...")
    bridge = VisionBridge(model_path=model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara; usa --sim.")

    slam = EKFSlam(0.0, 0.0, 0.0)
    grid = OccupancyGrid(size_m=8.0, resolution=0.05)
    trajectory = []
    output = slam.get_output()
    last_t = time.time()

    print("Presiona 'q' para salir y generar el croquis del mapa.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[aviso] Frame de cámara perdido.")
                continue

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            # TODO: sustituir por el módulo real de odometría/IMU.
            motion = MotionEstimate(delta_trans=0.0, delta_rot=0.0, dt=dt)

            detections = bridge.process(frame)
            # TODO: agregar aquí las lecturas ultrasónicas reales cuando
            # exista el hardware: fuse_observations(detections, us_readings)
            observations = fuse_observations(detections, [])

            output = slam.step(motion, observations)
            grid.update_from_scan((output.x, output.y, output.theta),
                                  slam.observations_for_map())
            trajectory.append((output.x, output.y))

            frame = annotate_frame(frame, detections, bridge.hfov, output)
            map_view = draw_map_view(grid, (output.x, output.y, output.theta),
                                     landmarks=slam.confirmed_landmarks(),
                                     size=240)
            frame[0:240, 0:240] = cv2.addWeighted(
                frame[0:240, 0:240], 0.15, map_view, 0.85, 0)

            cv2.imshow("SLAM - Camara + Mapa", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        finish(grid, output, trajectory, slam.confirmed_landmarks())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Sin cámara: cuarto simulado (sim_world.py)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice de la cámara (modo real)")
    parser.add_argument("--model", default=None,
                        help="Ruta a pesos YOLOv8-seg (por defecto, los de "
                             "vision/TT-NavIA)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Límite de ciclos en modo simulado")
    args = parser.parse_args()

    if args.sim:
        run_sim(max_steps=args.steps)
    else:
        run_camera(camera_index=args.camera, model_path=args.model)
