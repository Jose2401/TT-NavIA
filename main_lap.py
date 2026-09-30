"""
main_lap.py
MAIN SECUNDARIO — prueba en LAPTOP con interfaz gráfica.

Ejecuta el pipeline completo de los tres módulos integrados (SLAM +
Ray Casting/Mapeo por chunks + Visión) con el hardware disponible en la
laptop: SOLO la cámara (sin ultrasonido MB1232, sin odometría — igual
que hoy). Sirve para VER lo que en la Raspberry será salida auditiva:
qué se detecta, qué se mapea, en qué chunk/cuarto está el usuario y qué
recibiría el módulo de navegación.

    python main_lap.py                        # cámara 0, mapa nuevo/continuado
    python main_lap.py --camera 1
    python main_lap.py --video paseo.mp4      # video grabado
    python main_lap.py --sim                  # sin cámara: cuarto simulado
    python main_lap.py --map-dir maps/casa    # persistencia del mapa
    python main_lap.py --fresh                # ignora el mapa guardado
    python main_lap.py --hazards              # peligros de piso (SegFormer,
                                              #   lento sin GPU)

Controles (ventana activa):
    q  salir (guarda mapa + landmarks + croquis)
    m  guardar el mapa ahora
    en --sim: w/s/a/d mueven al usuario, p alterna piloto automático

Paneles:
    izquierda  cámara anotada (o vista del cuarto simulado)
    derecha    mapa global por chunks: pose, elipse de incertidumbre,
               landmarks, peligros (rojo), fronteras de chunk (morado)
    abajo      estado de navegación: d/riesgo por sector + alertas
"""
import argparse
import math
import os
import time

import numpy as np

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from occupancy_grid import OCC_THRESHOLD, HAZARD_MIN_HITS
from interfaces import MotionEstimate
from vision_bridge import fuse_observations, free_space_rays
from navigation_feed import build_navigation_frame, safety_alerts


# ----------------------------------------------------------------------
# Panel del mapa global por chunks
# ----------------------------------------------------------------------
def draw_chunk_map_panel(mgr, slam, output, events_log, size=520,
                         true_pose=None):
    import cv2
    lo, hz, (ox, oy) = mgr.stitched(include_disk=False)
    occ_prob = 1.0 / (1.0 + np.exp(-lo))
    img = (255 * (1 - occ_prob)).astype(np.uint8)
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    view[hz >= HAZARD_MIN_HITS] = (0, 0, 220)          # peligros en rojo

    h, w = view.shape[:2]
    scale = size / float(max(h, w))
    view = cv2.resize(view, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_NEAREST)
    view = cv2.flip(view, 0)                            # "arriba" = +y
    vh, vw = view.shape[:2]

    def to_px(wx, wy):
        gx = wx / mgr.resolution - ox
        gy = wy / mgr.resolution - oy
        return int(gx * scale), int(vh - 1 - gy * scale)

    # fronteras de chunk
    for (ci, cj) in mgr.known_chunk_indices():
        x0, y0 = to_px(ci * mgr.chunk_size_m, cj * mgr.chunk_size_m)
        x1, y1 = to_px((ci + 1) * mgr.chunk_size_m,
                       (cj + 1) * mgr.chunk_size_m)
        cv2.rectangle(view, (x0, y0), (x1, y1), (180, 100, 180), 1)
        chunk = mgr.chunks.get((ci, cj))
        tag = f"{ci},{cj}"
        if chunk is not None and chunk.room:
            tag += f" {chunk.room}"
        cv2.putText(view, tag, (min(x0, x1) + 3, max(y0, y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 100, 180), 1,
                    cv2.LINE_AA)

    # landmarks etiquetados
    for lm in slam.get_landmarks():
        px, py = to_px(lm["x"], lm["y"])
        if 0 <= px < vw and 0 <= py < vh:
            cv2.drawMarker(view, (px, py), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 8, 1)
            if lm["label"] not in ("eco_us", "obstaculo"):
                cv2.putText(view, lm["label"], (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255),
                            1, cv2.LINE_AA)

    if true_pose is not None:
        gx, gy = to_px(true_pose[0], true_pose[1])
        cv2.circle(view, (gx, gy), 5, (0, 180, 0), -1)

    # elipse de incertidumbre 2-sigma + pose estimada
    P = np.array(output.covariance)[:2, :2]
    vals, vecs = np.linalg.eigh(P)
    vals = np.maximum(vals, 0.0)
    px_per_m = scale / mgr.resolution
    axes = (max(2, int(2 * math.sqrt(vals[1]) * px_per_m)),
            max(2, int(2 * math.sqrt(vals[0]) * px_per_m)))
    angle = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    px, py = to_px(output.x, output.y)
    cv2.ellipse(view, (px, py), axes, angle, 0, 360, (0, 0, 200), 1,
                cv2.LINE_AA)
    cv2.circle(view, (px, py), 6, (0, 0, 255), -1)
    cv2.line(view, (px, py),
             (int(px + 22 * math.cos(output.theta)),
              int(py - 22 * math.sin(output.theta))), (0, 0, 255), 2)

    # eventos recientes (chunk/cuarto)
    for i, txt in enumerate(events_log[-3:]):
        cv2.putText(view, txt, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 80, 200), 1, cv2.LINE_AA)
    return view


def annotate_camera(frame, detections, hfov, yaw_ok, fps):
    """Camara anotada: verde = estático usable, rojo = móvil (no
    landmark), magenta = peligro de piso."""
    import cv2
    h, w = frame.shape[:2]
    for det in detections:
        col = int(w / 2.0 - (det.bearing / hfov) * w)
        if det.hazard:
            color = (255, 0, 255)
        elif det.moving:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.line(frame, (col, int(h * 0.30)), (col, int(h * 0.70)), color, 2)
        tag = det.label
        if det.range_est is not None:
            tag += f" {det.range_est:.1f}m"
        if det.moving:
            tag += " [movil]"
        if det.hazard:
            tag += " [PELIGRO PISO]"
        cv2.putText(frame, tag, (max(0, col - 60), int(h * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    status = "giro visual: OK" if yaw_ok else "giro visual: SIN TEXTURA"
    cv2.putText(frame, f"{status} | {fps:.1f} fps | q: salir y guardar",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_nav_bar(frame_nav, alerts, width, height=64):
    """Barra inferior: el estado que recibirá el módulo de navegación
    (d y riesgo por sector) + alertas de seguridad."""
    import cv2
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    risk_color = {0: (0, 200, 0), 1: (0, 200, 200),
                  2: (0, 140, 255), 3: (0, 0, 255)}

    def fmt(d):
        return "libre" if d >= 3.99 else f"{d:.2f}m"

    items = (("IZQ", frame_nav.d_left, frame_nav.risk_left),
             ("FRENTE", frame_nav.d_front, frame_nav.risk_front),
             ("DER", frame_nav.d_right, frame_nav.risk_right))
    seg = width // 3
    for i, (name, d, r) in enumerate(items):
        cv2.putText(bar, f"{name}: {fmt(d)} r={r}",
                    (i * seg + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    risk_color[r], 2, cv2.LINE_AA)
    if alerts:
        prio, msg = sorted(alerts)[0]
        color = (0, 0, 255) if prio == 0 else (0, 165, 255)
        cv2.putText(bar, msg, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, "sin alertas", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1,
                    cv2.LINE_AA)
    return bar


def describe_events(events, log):
    for ev in events:
        if ev["type"] == "chunk_change":
            estado = "CONOCIDO" if ev["known"] else "NUEVO"
            room = f" ({ev['room']})" if ev.get("room") else ""
            txt = f"Chunk {ev['chunk']} {estado}{room}"
        elif ev["type"] == "door_crossed":
            txt = f"Cruce de puerta -> {ev['room']}"
        elif ev["type"] == "room_recognized":
            txt = f"Cuarto reconocido: {ev['room']}"
        else:
            txt = str(ev)
        print(f"[evento] {txt}")
        log.append(txt)


# ----------------------------------------------------------------------
# Modo cámara (laptop): visión real, sin ultrasonido ni odometría
# ----------------------------------------------------------------------
def run_camera(args):
    import cv2
    from vision_bridge import (VisionBridge, VisualYawEstimator,
                               GroundHazardDetector)

    print("Cargando módulo de visión (YOLOv8-seg de paquete vision/)...")
    bridge = VisionBridge(model_path=args.model,
                          camera_profile=args.camera_profile,
                          cam_height_m=args.cam_height,
                          cam_pitch_deg=args.cam_pitch)
    yaw_est = VisualYawEstimator(hfov_deg=math.degrees(bridge.hfov))
    hazard_det = None
    if args.hazards:
        print("Cargando segmentación de escena (SegFormer ADE20K) para "
              "peligros de piso...")
        hazard_det = GroundHazardDetector(
            camera_profile=args.camera_profile,
            cam_height_m=args.cam_height, cam_pitch_deg=args.cam_pitch)

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {source}")

    slam, mgr = _setup_slam_and_map(args)
    trajectory = []
    output = slam.get_output()
    events_log = []
    last_hazards = []
    last_t = time.time()
    fps = 0.0
    frames = 0

    print("Mapeando con la cámara (sin ultrasonido/odometría: TODO "
          "hardware). 'q' para salir y guardar.")
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                print("[aviso] Frame de cámara perdido.")
                continue
            frames += 1
            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # 1) giro por flujo óptico; traslación 0 hasta tener odometría
            delta_rot, yaw_ok = yaw_est.update(frame)
            motion = MotionEstimate(0.0, delta_rot, dt)

            # 2) visión: objetos YOLO + peligros de piso cada N frames
            detections = bridge.process(frame)
            if hazard_det is not None and frames % args.hazard_every == 0:
                last_hazards = hazard_det.process(frame, detections)
            detections = detections + last_hazards

            # 3) SLAM + mapa por chunks (sin ultrasonido)
            observations = fuse_observations(detections, [])
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            events = mgr.update_position(pose, slam.get_landmarks())
            describe_events(events, events_log)
            trajectory.append((output.x, output.y))

            # 4) lo que recibirá navegación + alertas de seguridad
            nav = build_navigation_frame(
                slam, mgr, detections, max_range=args.max_range,
                chunk=mgr.current_chunk, room=mgr.current_room)
            alerts = safety_alerts(nav)
            for prio, msg in alerts:
                if prio == 0:
                    print(f"[ALERTA] {msg}")

            # 5) interfaz
            if not args.headless:
                cam_v = annotate_camera(frame.copy(), detections,
                                        bridge.hfov, yaw_ok, fps)
                map_v = draw_chunk_map_panel(mgr, slam, output, events_log,
                                             size=cam_v.shape[0])
                hud = _hud_line(output, mgr, slam)
                cv2.putText(cam_v, hud, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60),
                            2, cv2.LINE_AA)
                top = _hstack_pad(cam_v, map_v)
                bar = draw_nav_bar(nav, alerts, top.shape[1])
                cv2.imshow("NavIA - laptop: vision | mapa por chunks",
                           np.vstack([top, bar]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    n = mgr.save_all()
                    mgr.save_landmarks(slam.export_landmarks())
                    print(f"[mapa] guardados {n} chunks")
            elif frames % 30 == 0:
                print(f"[f={frames:04d}] {_hud_line(output, mgr, slam)} "
                      f"d=({nav.d_front:.1f},{nav.d_left:.1f},"
                      f"{nav.d_right:.1f})")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    _finish(args, mgr, slam, output, trajectory)


# ----------------------------------------------------------------------
# Modo simulado: cuarto sintético, mismo pipeline, chunks visibles
# ----------------------------------------------------------------------
def run_sim(args):
    from sim_world import (SimulatedWorld, SimulatedMotion,
                           SimulatedUltrasonicArray, simulated_vision)
    from sim_world import KeyboardUser

    if not args.headless:
        try:
            import cv2
        except ImportError:
            print("[aviso] OpenCV no disponible: modo headless.")
            args.headless = True
    if args.headless and args.max_frames is None:
        args.max_frames = 400

    world = SimulatedWorld.default_room()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    autopilot_driver = SimulatedMotion(start=(0.7, 0.7, 0.0))
    us_array = SimulatedUltrasonicArray(world)

    slam, mgr = _setup_slam_and_map(args, x0=0.7, y0=0.7)
    autopilot = True
    trajectory, gt_traj = [], []
    events_log = []
    output = slam.get_output()
    step = 0

    print("Modo simulado con chunks de "
          f"{mgr.chunk_size_m} m. wasd para moverte, p: autopiloto, "
          "q: salir y guardar.")
    while args.max_frames is None or step < args.max_frames:
        step += 1
        if autopilot:
            autopilot_driver.true_pose = list(user.true_pose)
            motion = autopilot_driver.get()
            user.true_pose = list(autopilot_driver.true_pose)
        else:
            import cv2
            forward = rot = 0.0
            key = cv2.waitKey(40) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                autopilot = True
                continue
            elif key == ord('w'):
                forward = 0.06
            elif key == ord('s'):
                forward = -0.06
            elif key == ord('a'):
                rot = math.radians(6)
            elif key == ord('d'):
                rot = -math.radians(6)
            motion = user.command(forward, rot)

        true_pose = tuple(user.true_pose)
        us_readings = us_array.read(true_pose)
        detections = simulated_vision(world, true_pose)
        observations = fuse_observations(detections, us_readings)

        output = slam.step(motion, observations)
        pose = (output.x, output.y, output.theta)
        mgr.update_from_scan(pose, slam.observations_for_map()
                             + free_space_rays(us_readings),
                             max_range=args.max_range)
        events = mgr.update_position(pose, slam.get_landmarks())
        describe_events(events, events_log)
        trajectory.append((output.x, output.y))
        gt_traj.append((true_pose[0], true_pose[1]))

        nav = build_navigation_frame(slam, mgr, detections,
                                     max_range=args.max_range,
                                     chunk=mgr.current_chunk,
                                     room=mgr.current_room,
                                     ultrasonic=us_readings)
        alerts = safety_alerts(nav)

        if not args.headless:
            import cv2
            map_v = draw_chunk_map_panel(mgr, slam, output, events_log,
                                         size=560, true_pose=true_pose)
            cv2.putText(map_v, _hud_line(output, mgr, slam), (8, map_v.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 1,
                        cv2.LINE_AA)
            bar = draw_nav_bar(nav, alerts, map_v.shape[1])
            cv2.imshow("NavIA - sim: mapa por chunks", np.vstack([map_v, bar]))
            if autopilot:
                key = cv2.waitKey(40) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    autopilot = False
        elif step % 50 == 0:
            err = math.hypot(output.x - true_pose[0],
                             output.y - true_pose[1])
            print(f"[t={step:04d}] {_hud_line(output, mgr, slam)} "
                  f"err={err:.2f}m")

    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    _finish(args, mgr, slam, output, trajectory, gt_traj)


# ----------------------------------------------------------------------
# Utilidades comunes
# ----------------------------------------------------------------------
def _setup_slam_and_map(args, x0=0.0, y0=0.0):
    map_dir = None if args.no_save else args.map_dir
    if map_dir and args.fresh and os.path.isdir(map_dir):
        import shutil
        shutil.rmtree(map_dir)
        print(f"[mapa] {map_dir} borrado (--fresh)")
    mgr = ChunkedMapManager(map_dir=map_dir,
                            chunk_size_m=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(x0, y0, 0.0)
    saved = mgr.load_landmarks()
    if saved:
        slam.import_landmarks(saved)
        print(f"[mapa] {len(saved)} landmarks restaurados de {map_dir} "
              "(re-localización activa)")
    if mgr.known_chunk_indices():
        print(f"[mapa] mapa conocido: {len(mgr.known_chunk_indices())} "
              f"chunks en {map_dir}")
    return slam, mgr


def _hud_line(output, mgr, slam):
    room = f" cuarto={mgr.current_room}" if mgr.current_room else ""
    return (f"x={output.x:+.2f} y={output.y:+.2f} "
            f"th={math.degrees(output.theta):+.0f}deg "
            f"conf={output.confidence:.2f} "
            f"chunk={mgr.current_chunk}{room} "
            f"lm={len(slam.confirmed_landmarks())}")


def _hstack_pad(a, b):
    import cv2
    h = max(a.shape[0], b.shape[0])

    def pad(img):
        if img.shape[0] == h:
            return img
        out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
        out[:img.shape[0]] = img
        return out
    return np.hstack([pad(a), pad(b)])


def _finish(args, mgr, slam, output, trajectory, gt_traj=None):
    if not args.no_save:
        n = mgr.save_all()
        mgr.save_landmarks(slam.export_landmarks())
        print(f"[mapa] {n} chunks + landmarks guardados en {mgr.map_dir}")
    path = mgr.render_croquis(
        pose=(output.x, output.y, output.theta),
        trajectory=trajectory, gt_trajectory=gt_traj,
        landmarks=slam.confirmed_landmarks(),
        path=args.croquis)
    print(f"Croquis global guardado en: {path}")
    print(f"Pose final: x={output.x:.2f} y={output.y:.2f} "
          f"theta={math.degrees(output.theta):.1f} deg "
          f"(confianza={output.confidence:.2f})")


def build_parser():
    p = argparse.ArgumentParser(
        description="Main de prueba en laptop (GUI) del pipeline "
                    "SLAM + mapeo por chunks + visión")
    p.add_argument("--sim", action="store_true",
                   help="cuarto simulado en vez de cámara")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--camera-profile", default="laptop",
                   choices=["laptop", "picam3wide"])
    p.add_argument("--cam-height", type=float, default=1.55,
                   help="altura de la cámara (m) para plano de piso")
    p.add_argument("--cam-pitch", type=float, default=10.0,
                   help="inclinación de la cámara hacia abajo (grados)")
    p.add_argument("--hazards", action="store_true",
                   help="detección de peligros de piso (SegFormer)")
    p.add_argument("--hazard-every", type=int, default=5,
                   help="ejecutar segmentación cada N frames")
    p.add_argument("--map-dir", default=os.path.join("maps", "default"),
                   help="carpeta de persistencia del mapa por chunks")
    p.add_argument("--no-save", action="store_true",
                   help="no persistir el mapa (solo memoria)")
    p.add_argument("--fresh", action="store_true",
                   help="borrar el mapa guardado y empezar de cero")
    p.add_argument("--chunk-size", type=float, default=8.0,
                   help="lado del chunk en metros")
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0,
                   help="alcance máximo de los rayos del mapa")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--croquis", default="croquis_chunks.png")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.sim:
        run_sim(args)
    else:
        run_camera(args)
