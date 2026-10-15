"""
main_nav_pi.py — navegación completa en la raspberry (headless, audio).

igual que main_nav_lap.py pero sin gui. ojo: el modelo drl corre con el
backend de numpy puro (models/nav_ppo_policy.npz), la pi no ocupa
stable-baselines3 (torch sí, por yolo). comandos por stdin mientras no
exista el módulo de voz/pln (TODO: ese módulo solo debe llamar
navigator.handle_command con la intención detectada).

    python main_nav_pi.py --map-dir maps/casa --model-nav models/nav_ppo
    ctrl+c guarda mapa y sale.
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
from main_pi import Mb1232Array, VoiceOutput
from odometry import make_motion_estimator


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
    p.add_argument("--no-imu", action="store_true",
                   help="no usar la IMU BNO055 (solo flujo óptico)")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x28)
    p.add_argument("--imu-bus", type=int, default=1)
    p.add_argument("--step-length", type=float, default=0.68)
    args = p.parse_args()

    import cv2
    from vision_bridge import VisionBridge, GroundHazardDetector

    voice = VoiceOutput()
    voice.say("Sistema listo. Diga su destino.", priority=2)

    bridge = VisionBridge(model_path=args.model, camera_profile="picam3wide")
    hazard_det = GroundHazardDetector(camera_profile="picam3wide") \
        if args.hazards else None
    motion_est = make_motion_estimator(
        use_imu=not args.no_imu, imu_bus=args.imu_bus,
        imu_addr=args.imu_addr, step_length_m=args.step_length,
        hfov_deg=math.degrees(bridge.hfov),
        say=lambda m: voice.say(m, priority=2))
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

            # comandos: stdin hoy, módulo de voz/pln mañana
            try:
                while True:
                    for m in navigator.handle_command(cmd_q.get_nowait(),
                                                      nav_frame):
                        voice.say(m.text, priority=m.priority)
            except queue.Empty:
                pass

            motion, odo = motion_est.update(frame, dt)
            # pitch real de la imu -> geometría de plano de piso
            if odo["pitch_deg"] is not None:
                bridge.cam_pitch_deg = odo["pitch_deg"]
                if hazard_det is not None:
                    hazard_det.cam_pitch_deg = odo["pitch_deg"]
            detections = bridge.process(frame, ego_yaw=motion.delta_rot)
            if hazard_det is not None and frames % args.hazard_every == 0:
                last_hazards = hazard_det.process(frame, detections)
            detections = detections + last_hazards

            us_readings = ultrasonic.read()
            observations = fuse_observations(detections, us_readings)
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            mgr.clear_footprint(output.x, output.y)
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

            # seguridad primero (p0 interrumpe); una alerta crítica fuerza
            # recalcular de inmediato
            alerts = safety_alerts(nav_frame)
            for prio, msg in alerts:
                voice.say(msg, priority=prio,
                          repeat_after_s=2.0 if prio == 0 else 5.0)
            for m in navigator.update(nav_frame, mgr, alerts):
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
