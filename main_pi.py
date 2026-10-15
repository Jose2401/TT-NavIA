"""
main_pi.py — main principal en la raspberry (sin pantalla, todo es audio).

mismo pipeline que main_lap.py pero headless. ojo: el hardware que falta
queda marcado con TODO aquí mismo (ultrasonido MB1232 en Mb1232Array,
voz real en VoiceOutput.say); la imu bno055 ya está en odometry.py con
fallback a flujo óptico.

    python main_pi.py --map-dir maps/casa
    python main_pi.py --hazards --hazard-every 10
    ctrl+c guarda mapa + landmarks y sale.
"""
import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuse_observations
from navigation_feed import build_navigation_frame, safety_alerts


class Mb1232Array:
    """par de MB1232 (izq/der). TODO hardware: smbus2, write 0x51 ->
    esperar ~100ms -> leer 2 bytes (cm); direcciones y montaje según
    TT_2026_B045 §7.6. mientras, read() da [] y jala solo con cámara."""

    LEFT_BEARING = math.radians(90)
    RIGHT_BEARING = math.radians(-90)
    MAX_RANGE_M = 7.65

    def __init__(self, addr_left=0x20, addr_right=0x21, bus=1):
        self.addr_left = addr_left
        self.addr_right = addr_right
        self.available = False   # TODO: detectar el bus i2c real

    def read(self):
        """-> list[UltrasonicReading]. TODO hardware."""
        return []


class VoiceOutput:
    """TODO: conectar el tts real del equipo (pyttsx3 con cola
    priorizada 0/1/2). por ahora imprime con prefijo [VOZ]."""

    def __init__(self):
        self._last = {}

    def say(self, msg, priority=2, repeat_after_s=4.0):
        # no repetir la misma frase en ráfaga (el tts real lo hará solo)
        now = time.time()
        if now - self._last.get(msg, 0.0) < repeat_after_s:
            return
        self._last[msg] = now
        print(f"[VOZ p{priority}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Main Raspberry Pi (headless)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--cam-height", type=float, default=1.55)
    p.add_argument("--cam-pitch", type=float, default=10.0)
    p.add_argument("--hazards", action="store_true",
                   help="peligros de piso (SegFormer; costoso en la Pi)")
    p.add_argument("--hazard-every", type=int, default=10)
    p.add_argument("--map-dir", default=os.path.join("maps", "default"))
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-every", type=float, default=30.0,
                   help="autoguardado del mapa cada N segundos")
    p.add_argument("--no-imu", action="store_true",
                   help="no usar la IMU BNO055 (solo flujo óptico)")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x28,
                   help="dirección I2C del BNO055 (0x28 o 0x29)")
    p.add_argument("--imu-bus", type=int, default=1)
    p.add_argument("--step-length", type=float, default=0.68,
                   help="longitud de paso (m) para el avance por pasos; "
                        "calibrar caminando una distancia conocida")
    args = p.parse_args()

    import cv2
    from vision_bridge import VisionBridge, GroundHazardDetector
    from odometry import make_motion_estimator

    voice = VoiceOutput()
    voice.say("Iniciando sistema de navegacion", priority=2)

    bridge = VisionBridge(model_path=args.model,
                          camera_profile="picam3wide",
                          cam_height_m=args.cam_height,
                          cam_pitch_deg=args.cam_pitch)
    hazard_det = GroundHazardDetector(
        camera_profile="picam3wide", cam_height_m=args.cam_height,
        cam_pitch_deg=args.cam_pitch) if args.hazards else None
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
        voice.say("Mapa conocido cargado", priority=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    last_t = time.time()
    last_save = last_t
    last_hazards = []
    frames = 0
    nav = None

    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now

            motion, odo = motion_est.update(frame, dt)
            # pitch real de la imu -> bridge: la geometría de plano de piso
            # usa el ángulo verdadero, no el fijo de calibración
            if odo["pitch_deg"] is not None:
                bridge.cam_pitch_deg = odo["pitch_deg"]
                if hazard_det is not None:
                    hazard_det.cam_pitch_deg = odo["pitch_deg"]
            detections = bridge.process(frame, ego_yaw=motion.delta_rot)
            if hazard_det is not None and frames % args.hazard_every == 0:
                last_hazards = hazard_det.process(frame, detections)
            detections = detections + last_hazards

            us_readings = ultrasonic.read()          # [] hasta tener hw
            observations = fuse_observations(detections, us_readings)
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            mgr.clear_footprint(output.x, output.y)
            events = mgr.update_position(pose, slam.get_landmarks())

            for ev in events:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voice.say("Zona nueva", priority=2)
                elif ev["type"] == "room_recognized":
                    voice.say(f"Estas en {ev['room']}", priority=1)
                elif ev["type"] == "door_crossed":
                    voice.say("Cruzaste una puerta", priority=1)

            # este NavigationFrame es la entrada del módulo drl
            nav = build_navigation_frame(
                slam, mgr, detections, max_range=args.max_range,
                chunk=mgr.current_chunk, room=mgr.current_room,
                ultrasonic=us_readings)

            for prio, msg in safety_alerts(nav):
                voice.say(msg, priority=prio,
                          repeat_after_s=2.0 if prio == 0 else 5.0)

            if now - last_save > args.save_every:
                mgr.save_all()
                mgr.save_landmarks(slam.export_landmarks())
                last_save = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        n = mgr.save_all()
        mgr.save_landmarks(slam.export_landmarks())
        print(f"\n[mapa] {n} chunks + landmarks guardados en {args.map_dir}")
        voice.say("Sistema detenido", priority=2)


if __name__ == "__main__":
    main()
