"""
main_nav_lap.py
NAVEGACIÓN COMPLETA en LAPTOP (GUI) — el sistema entero trabajando junto:

    SLAM + mapeo por chunks + visión  ->  NavigationFrame
    NavigationFrame + comandos        ->  Navigator (modelo DRL)
    Navigator                         ->  indicaciones una a una

Como en la laptop no hay voz, los comandos se escriben en la TERMINAL
(hilo aparte) y las indicaciones se muestran en pantalla y consola.

    python main_nav_lap.py --sim                  # cuarto simulado
    python main_nav_lap.py                        # cámara real
    python main_nav_lap.py --model-nav models/nav_ppo

Comandos de ejemplo (escribir en la terminal y Enter):
    ve a la silla        |  llevame a la salida  |  punto 3.5 2.0
    pausa  |  reanuda  |  detener  |  ruta alternativa  |  estado

En modo --sim el usuario se mueve SOLO siguiendo las indicaciones del
navegador (autopiloto de obediencia): es la validación de que las
indicaciones, una a una, llevan al usuario al destino esquivando
obstáculos. Con 'o' se alterna a control manual (wasd).
"""
import argparse
import math
import os
import queue
import threading
import time

import numpy as np

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from interfaces import MotionEstimate
from vision_bridge import fuse_observations, free_space_rays
from navigation_feed import build_navigation_frame, safety_alerts
from navigation.navigator import Navigator, FORWARD_STEP, TURN_STEP
from main_lap import (draw_chunk_map_panel, draw_nav_bar, annotate_camera,
                      describe_events, _setup_slam_and_map, _hud_line,
                      _hstack_pad)


def start_command_thread(q):
    def loop():
        while True:
            try:
                line = input()
            except EOFError:
                return
            q.put(line)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def draw_nav_overlay(view, navigator, mgr, to_px_scale=None):
    """Marca el destino y el waypoint actual sobre el panel del mapa."""
    import cv2
    st = navigator.status
    if st.goal is None:
        return view
    lo, hz, (ox, oy) = mgr.stitched(include_disk=False)
    h, w = lo.shape
    vh, vw = view.shape[:2]
    scale = min(vh / h, vw / w) if h and w else 1.0

    def to_px(wx, wy):
        gx = wx / mgr.resolution - ox
        gy = wy / mgr.resolution - oy
        return int(gx * scale), int(vh - 1 - gy * scale)

    gx, gy = to_px(*st.goal)
    cv2.drawMarker(view, (gx, gy), (255, 0, 200), cv2.MARKER_STAR, 18, 2)
    label = st.goal_label or "destino"
    cv2.putText(view, label, (gx + 8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 200), 1, cv2.LINE_AA)
    return view


def print_msgs(msgs):
    for m in msgs:
        print(f"[NAV p{m.priority}] {m.text}", flush=True)


# ----------------------------------------------------------------------
# Modo simulado: el usuario OBEDECE las indicaciones del navegador
# ----------------------------------------------------------------------
def run_sim(args):
    if not args.headless:
        import cv2
    from sim_world import (SimulatedWorld, SimulatedUltrasonicArray,
                           simulated_vision)
    from sim_world import KeyboardUser
    if args.headless and args.max_frames is None:
        args.max_frames = 500

    world = SimulatedWorld.default_room()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    us_array = SimulatedUltrasonicArray(world)
    slam, mgr = _setup_slam_and_map(args, x0=0.7, y0=0.7)
    # La cadencia decisión/paso reproduce el entrenamiento: una decisión
    # por ~2 frames de simulación, avanzando FORWARD_STEP/2 por frame.
    navigator = Navigator(args.model_nav, decide_every_s=0.0,
                          repeat_s=2.5)

    cmd_q = queue.Queue()
    start_command_thread(cmd_q)
    print(__doc__)
    print("[NAV p2] Sistema listo. Diga su destino. "
          "(ej: 've a la silla', 'punto 5 3')")

    obey = True          # obedecer indicaciones del navegador
    events_log = []
    trajectory = []
    output = slam.get_output()
    nav_frame = None
    step = 0

    while args.max_frames is None or step < args.max_frames:
        step += 1

        # ---- comandos desde la terminal ----
        try:
            while True:
                line = cmd_q.get_nowait()
                print_msgs(navigator.handle_command(line, nav_frame))
        except queue.Empty:
            pass

        # ---- movimiento: obedecer la última acción del navegador ----
        forward = rot = 0.0
        if args.headless:
            key = 255
            time.sleep(0.0)          # headless: sin cadencia de teclado
        else:
            import cv2
            key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            obey = not obey
            print(f"[modo] {'obedecer navegador' if obey else 'manual wasd'}")
        elif not obey:
            if key == ord('w'):
                forward = FORWARD_STEP / 2
            elif key == ord('s'):
                forward = -FORWARD_STEP / 2
            elif key == ord('a'):
                rot = TURN_STEP / 2
            elif key == ord('d'):
                rot = -TURN_STEP / 2
        elif navigator.status.session == "activo" \
                and navigator.status.last_action is not None:
            act = navigator.status.last_action
            if act == 0:
                forward = FORWARD_STEP / 2      # medio paso por frame
            elif act == 1:
                rot = TURN_STEP / 2
            elif act == 2:
                rot = -TURN_STEP / 2
        motion = user.command(forward, rot)

        # ---- pipeline SLAM + mapa (idéntico a main_lap) ----
        true_pose = tuple(user.true_pose)
        us_readings = us_array.read(true_pose)
        detections = simulated_vision(world, true_pose)
        observations = fuse_observations(detections, us_readings)
        output = slam.step(motion, observations)
        pose = (output.x, output.y, output.theta)
        mgr.update_from_scan(pose, slam.observations_for_map()
                             + free_space_rays(us_readings),
                             max_range=args.max_range)
        describe_events(mgr.update_position(pose, slam.get_landmarks()),
                        events_log)
        trajectory.append((output.x, output.y))

        nav_frame = build_navigation_frame(
            slam, mgr, detections, max_range=args.max_range,
            chunk=mgr.current_chunk, room=mgr.current_room,
            ultrasonic=us_readings)

        # ---- navegación: decisión + indicaciones + alertas ----
        print_msgs(navigator.update(nav_frame, mgr))
        for prio, msg in safety_alerts(nav_frame):
            if prio == 0:
                print(f"[ALERTA p0] {msg}")

        # ---- GUI ----
        if not args.headless:
            import cv2
            view = draw_chunk_map_panel(mgr, slam, output, events_log,
                                        size=560, true_pose=true_pose)
            view = draw_nav_overlay(view, navigator, mgr)
            st = navigator.status
            estado = f"sesion={st.session}"
            if st.goal_label:
                estado += f" destino={st.goal_label}"
            if st.d_goal:
                estado += f" faltan={st.d_goal:.1f}m"
            cv2.putText(view, estado, (8, view.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(view, _hud_line(output, mgr, slam),
                        (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 60, 60), 1, cv2.LINE_AA)
            bar = draw_nav_bar(nav_frame, safety_alerts(nav_frame),
                               view.shape[1])
            cv2.imshow("NavIA - navegacion DRL (sim)",
                       np.vstack([view, bar]))

    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    if not args.no_save:
        mgr.save_all()
        mgr.save_landmarks(slam.export_landmarks())


# ----------------------------------------------------------------------
# Modo cámara (laptop real, sin ultrasonido/odometría)
# ----------------------------------------------------------------------
def run_camera(args):
    import cv2
    from vision_bridge import VisionBridge, VisualYawEstimator

    print("Cargando módulo de visión (YOLOv8-seg)...")
    bridge = VisionBridge(model_path=args.model,
                          camera_profile=args.camera_profile)
    yaw_est = VisualYawEstimator(hfov_deg=math.degrees(bridge.hfov))
    slam, mgr = _setup_slam_and_map(args)
    navigator = Navigator(args.model_nav)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video")

    cmd_q = queue.Queue()
    start_command_thread(cmd_q)
    print("[NAV p2] Sistema listo. Diga su destino.")

    events_log = []
    output = slam.get_output()
    nav_frame = None
    last_t = time.time()
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                continue
            frames += 1
            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            try:
                while True:
                    print_msgs(navigator.handle_command(cmd_q.get_nowait(),
                                                        nav_frame))
            except queue.Empty:
                pass

            delta_rot, yaw_ok = yaw_est.update(frame)
            motion = MotionEstimate(0.0, delta_rot, dt)
            detections = bridge.process(frame)
            observations = fuse_observations(detections, [])
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            describe_events(mgr.update_position(pose, slam.get_landmarks()),
                            events_log)

            nav_frame = build_navigation_frame(
                slam, mgr, detections, max_range=args.max_range,
                chunk=mgr.current_chunk, room=mgr.current_room)
            print_msgs(navigator.update(nav_frame, mgr))

            if not args.headless:
                cam_v = annotate_camera(frame.copy(), detections,
                                        bridge.hfov, yaw_ok, 1.0 / dt)
                map_v = draw_chunk_map_panel(mgr, slam, output, events_log,
                                             size=cam_v.shape[0])
                map_v = draw_nav_overlay(map_v, navigator, mgr)
                top = _hstack_pad(cam_v, map_v)
                bar = draw_nav_bar(nav_frame, safety_alerts(nav_frame),
                                   top.shape[1])
                cv2.imshow("NavIA - navegacion DRL (camara)",
                           np.vstack([top, bar]))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if not args.no_save:
            mgr.save_all()
            mgr.save_landmarks(slam.export_landmarks())


def build_parser():
    p = argparse.ArgumentParser(description="Navegación DRL en laptop (GUI)")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--model-nav", default="models/nav_ppo",
                   help="prefijo del modelo de navegación entrenado")
    p.add_argument("--camera-profile", default="laptop",
                   choices=["laptop", "picam3wide"])
    p.add_argument("--map-dir", default=os.path.join("maps", "default"))
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.sim:
        run_sim(args)
    else:
        run_camera(args)
