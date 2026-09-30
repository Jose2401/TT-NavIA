"""
main_nav_pi.py
NAVEGACIÓN COMPLETA en la RASPBERRY PI (headless, salida auditiva).

Mismo pipeline que main_nav_lap.py pero sin GUI. El modelo de
navegación corre con el backend de NUMPY PURO (models/nav_ppo_policy.npz):
la Raspberry NO necesita stable-baselines3 (torch sí, por YOLO).

Entrada de comandos: por ahora stdin (una línea = un comando); cuando
exista el módulo de voz/PLN, este solo tiene que llamar
`navigator.handle_command(NavCommand(...))` con la intención detectada
(destino/pausa/reanudar/detener/ruta_alternativa) — el formato ya está
definido en navigation/navigator.py.

Salida: NavMessage con prioridad (0 alerta interrumpe, 1 navegación,
2 confirmación), impresos como [VOZ pN]; el módulo de voz real (pyttsx3
con cola priorizada) los reproducirá tal cual.

    python main_nav_pi.py --map-dir maps/casa --model-nav models/nav_ppo
    Ctrl+C -> guarda mapa y termina.
"""
import argparse
import math
import os
import queue
import threading
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuse_observations
from navigation_feed import build_navigation_frame, safety_alerts
from navigation.navigator import Navigator
from main_pi import Mb1232Array, VoiceOutput, read_motion


def start_command_thread(q):
    def loop():
        while True:
            try:
                q.put(input())
            except EOFError:
                return
    threading.Thread(target=loop, daemon=True).start()


def main():
    p = argparse.ArgumentParser(description="Navegación DRL en Raspberry Pi")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--model-nav", default="models/nav_ppo")
    p.add_argument("--hazards", action="store_true")
    p.add_argument("--hazard-every", type=int, default=10)
    p.add_argument("--map-dir", default=os.path.join("maps", "default"))
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-every", type=float, default=30.0)
    args = p.parse_args()

    import cv2
    from vision_bridge import (VisionBridge, VisualYawEstimator,
                               GroundHazardDetector)

    voice = VoiceOutput()
    voice.say("Sistema listo. Diga su destino.", priority=2)

    bridge = VisionBridge(model_path=args.model, camera_profile="picam3wide")
    yaw_est = VisualYawEstimator(hfov_deg=math.degrees(bridge.hfov))
    hazard_det = GroundHazardDetector(camera_profile="picam3wide") \
        if args.hazards else None
    ultrasonic = Mb1232Array()

    mgr = ChunkedMapManager(map_dir=args.map_dir,
                            chunk_size_m=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    saved = mgr.load_landmarks()
    if saved:
        slam.import_landmarks(saved)
        voice.say("Mapa conocido cargado.", priority=2)

    navigator = Navigator(args.model_nav)
    cmd_q = queue.Queue()
    start_command_thread(cmd_q)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    last_t = time.time()
    last_save = last_t
    last_hazards = []
    nav_frame = None
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            # comandos (stdin hoy; módulo de voz/PLN mañana)
            try:
                while True:
                    for m in navigator.handle_command(cmd_q.get_nowait(),
                                                      nav_frame):
                        voice.say(m.text, priority=m.priority)
            except queue.Empty:
                pass

            motion, _ = read_motion(yaw_est, frame, dt)
            detections = bridge.process(frame)
            if hazard_det is not None and frames % args.hazard_every == 0:
                last_hazards = hazard_det.process(frame, detections)
            detections = detections + last_hazards

            us_readings = ultrasonic.read()
            observations = fuse_observations(detections, us_readings)
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            events = mgr.update_position(pose, slam.get_landmarks())
            for ev in events:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voice.say("Zona nueva.", priority=2)
                elif ev["type"] == "room_recognized":
                    voice.say(f"Está en {ev['room']}.", priority=1)

            nav_frame = build_navigation_frame(
                slam, mgr, detections, max_range=args.max_range,
                chunk=mgr.current_chunk, room=mgr.current_room,
                ultrasonic=us_readings)

            # seguridad primero (prioridad 0 interrumpe), luego navegación
            for prio, msg in safety_alerts(nav_frame):
                voice.say(msg, priority=prio,
                          repeat_after_s=2.0 if prio == 0 else 5.0)
            for m in navigator.update(nav_frame, mgr):
                voice.say(m.text, priority=m.priority)

            if now - last_save > args.save_every:
                mgr.save_all()
                mgr.save_landmarks(slam.export_landmarks())
                last_save = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        mgr.save_all()
        mgr.save_landmarks(slam.export_landmarks())
        voice.say("Sistema detenido.", priority=2)


if __name__ == "__main__":
    main()
