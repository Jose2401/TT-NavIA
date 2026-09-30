"""
demo_mapa_interactivo.py
Mapeo en TIEMPO REAL sin ningún sensor físico (ni cámara, ni ultrasonido,
ni IMU): todo se genera desde el cuarto simulado (sim_world.py), pero el
pipeline SLAM -> mapa es EXACTAMENTE el mismo que con hardware real.

Sirve para ver, en vivo, cómo el módulo cumple su tarea: mientras "el
usuario" camina por el cuarto, la ventana muestra el mapa de ocupación
construyéndose, la pose estimada (x, y) y hacia dónde mira (theta), la
elipse de incertidumbre del EKF, los landmarks descubiertos y la
confianza.

Uso:
    python demo_mapa_interactivo.py              # tú controlas al usuario
    python demo_mapa_interactivo.py --auto       # recorrido automático
    python demo_mapa_interactivo.py --auto --steps 300 --headless
                                                 # sin ventana (pruebas)

Controles (con la ventana activa):
    w / s   avanzar / retroceder
    a / d   girar a la izquierda / derecha
    p       alternar piloto automático (recorrido por waypoints)
    q       salir y generar el croquis final (croquis_interactivo.png)

Leyenda de la ventana:
    - Rojo: pose ESTIMADA por el SLAM (punto + flecha de orientación)
    - Elipse roja: incertidumbre 2-sigma de la posición (de la P del EKF)
    - Verde: pose REAL del simulador (para comparar a ojo la precisión)
    - Naranja: landmarks confirmados (con su etiqueta)
    - Gris -> blanco/negro: celdas desconocidas -> libres/ocupadas
"""
import argparse
import math
import random
import time

import numpy as np

from ekf_slam import EKFSlam, wrap_angle
from occupancy_grid import OccupancyGrid
from interfaces import MotionEstimate
from sim_world import (SimulatedWorld, SimulatedMotion, KeyboardUser,
                       SimulatedUltrasonicArray, simulated_vision)
from vision_bridge import fuse_observations, free_space_rays

STEP_DIST = 0.06            # m por pulsación de avance
STEP_ROT = math.radians(6)  # rad por pulsación de giro
ODOM_SIGMA_TRANS = 0.006    # ruido de la "odometría" simulada
ODOM_SIGMA_ROT = math.radians(0.5)
SCALE = 3                   # px por celda en la ventana


def draw_view(grid, slam, output, true_pose, autopilot):
    """Panel principal: mapa de ocupación + poses + landmarks + HUD."""
    import cv2

    occ_prob = 1.0 / (1.0 + np.exp(-grid.log_odds))
    img = (255 * (1 - occ_prob)).astype(np.uint8)
    img = np.flipud(img)                       # "arriba" = +y
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, (grid.n * SCALE, grid.n * SCALE),
                      interpolation=cv2.INTER_NEAREST)

    def to_px(wx, wy):
        gx, gy = grid.world_to_grid(wx, wy)
        return gx * SCALE, (grid.n - 1 - gy) * SCALE

    # Landmarks confirmados, con etiqueta
    for lm in slam.get_landmarks():
        px, py = to_px(lm["x"], lm["y"])
        cv2.drawMarker(view, (px, py), (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 10, 2)
        if lm["label"] not in ("eco_us", "obstaculo"):
            cv2.putText(view, lm["label"], (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1,
                        cv2.LINE_AA)

    # Pose real (verde), para comparar
    gx, gy = to_px(true_pose[0], true_pose[1])
    cv2.circle(view, (gx, gy), 6, (0, 180, 0), -1)
    cv2.line(view, (gx, gy),
             (int(gx + 20 * math.cos(true_pose[2])),
              int(gy - 20 * math.sin(true_pose[2]))), (0, 180, 0), 2)

    # Elipse de incertidumbre 2-sigma (posición) de la P del EKF
    P = np.array(output.covariance)[:2, :2]
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, 0.0)
    axes = (max(2, int(2 * math.sqrt(vals[1]) / grid.resolution * SCALE)),
            max(2, int(2 * math.sqrt(vals[0]) / grid.resolution * SCALE)))
    angle = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ex, ey = to_px(output.x, output.y)
    cv2.ellipse(view, (ex, ey), axes, angle, 0, 360, (0, 0, 200), 1,
                cv2.LINE_AA)

    # Pose estimada (rojo)
    cv2.circle(view, (ex, ey), 6, (0, 0, 255), -1)
    cv2.line(view, (ex, ey),
             (int(ex + 20 * math.cos(output.theta)),
              int(ey - 20 * math.sin(output.theta))), (0, 0, 255), 2)

    # HUD
    lines = [
        f"pose estimada: x={output.x:+.2f} m  y={output.y:+.2f} m  "
        f"theta={math.degrees(output.theta):+.0f} grados",
        f"confianza={output.confidence:.2f}   "
        f"landmarks={len(slam.confirmed_landmarks())}   "
        f"modo={'AUTO (p: manual)' if autopilot else 'MANUAL wasd (p: auto)'}",
        "rojo: estimado | verde: real | q: croquis y salir",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(view, txt, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 60, 60), 1, cv2.LINE_AA)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="iniciar en piloto automático")
    parser.add_argument("--steps", type=int, default=None,
                        help="terminar tras N ciclos (útil con --auto)")
    parser.add_argument("--headless", action="store_true",
                        help="sin ventana (solo consola + croquis final)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    headless = args.headless
    if not headless:
        try:
            import cv2
        except ImportError:
            print("[aviso] OpenCV no disponible: se continúa en modo headless.")
            headless = True
    if headless and args.steps is None:
        args.steps = 300

    world = SimulatedWorld.default_room()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    autopilot_driver = SimulatedMotion(start=(0.7, 0.7, 0.0))
    us_array = SimulatedUltrasonicArray(world)

    slam = EKFSlam(x0=0.7, y0=0.7, theta0=0.0)
    grid = OccupancyGrid(size_m=8.0, resolution=0.05, origin=(1.0, 1.0))

    autopilot = args.auto or headless
    trajectory, gt_traj = [], []
    output = slam.get_output()
    step = 0
    t_report = time.time()

    if not headless:
        import cv2
        print(__doc__)

    while args.steps is None or step < args.steps:
        step += 1

        # ---- 1) movimiento (manual o autopiloto) -> MotionEstimate ----
        if autopilot:
            autopilot_driver.true_pose = list(user.true_pose)
            motion = autopilot_driver.get()
            user.true_pose = list(autopilot_driver.true_pose)
        else:
            forward = rot = 0.0
            key = cv2.waitKey(40) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                autopilot = True
                continue
            elif key == ord('w'):
                forward = STEP_DIST
            elif key == ord('s'):
                forward = -STEP_DIST
            elif key == ord('a'):
                rot = STEP_ROT
            elif key == ord('d'):
                rot = -STEP_ROT
            motion = user.command(forward, rot)

        true_pose = tuple(user.true_pose)

        # ---- 2) "sensores" (simulados: no se necesita hardware) ----
        us_readings = us_array.read(true_pose)
        vision_dets = simulated_vision(world, true_pose)
        observations = fuse_observations(vision_dets, us_readings)

        # ---- 3) ciclo SLAM + mapa (idéntico al sistema real) ----
        output = slam.step(motion, observations)
        grid.update_from_scan((output.x, output.y, output.theta),
                              slam.observations_for_map()
                              + free_space_rays(us_readings))
        trajectory.append((output.x, output.y))
        gt_traj.append((true_pose[0], true_pose[1]))

        # ---- 4) presentación ----
        if not headless:
            view = draw_view(grid, slam, output, true_pose, autopilot)
            cv2.imshow("SLAM - mapa en tiempo real (sin sensores fisicos)",
                       view)
            if autopilot:
                key = cv2.waitKey(40) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    autopilot = False
        elif step % 50 == 0:
            err = math.hypot(output.x - true_pose[0], output.y - true_pose[1])
            print(f"[t={step:04d}] est=({output.x:.2f},{output.y:.2f},"
                  f"{math.degrees(output.theta):+6.1f} deg) err={err:.2f} m "
                  f"conf={output.confidence:.2f} "
                  f"landmarks={len(slam.confirmed_landmarks())}")
            t_report = time.time()

    if not headless:
        import cv2
        cv2.destroyAllWindows()

    path = grid.render_croquis(
        pose=(output.x, output.y, output.theta),
        trajectory=trajectory, gt_trajectory=gt_traj,
        landmarks=slam.confirmed_landmarks(),
        path="croquis_interactivo.png")
    print(f"\nCroquis guardado en: {path}")
    print(f"Pose final estimada: x={output.x:.2f} m, y={output.y:.2f} m, "
          f"theta={math.degrees(output.theta):.1f} grados "
          f"(confianza={output.confidence:.2f})")
    err = math.hypot(output.x - user.true_pose[0],
                     output.y - user.true_pose[1])
    print(f"Error final vs pose real del simulador: {err:.2f} m")


if __name__ == "__main__":
    main()
