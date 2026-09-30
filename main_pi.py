"""
main_pi.py
MAIN PRINCIPAL — Raspberry Pi (sin interfaz gráfica).

En el dispositivo final NO hay pantalla: toda la salida al usuario es
auditiva. Este main corre el mismo pipeline que main_lap.py (validado
antes en la laptop con GUI) y deja los puntos de conexión del hardware
que aún no existe claramente marcados:

    - Ultrasonido: 2x MB1232 I2CXL-MaxSonar por I2C (0x70 re-direccionados
      a 0x20/0x21 según el diseño), montados a IZQUIERDA y DERECHA.
      TODO en Mb1232Array.read().
    - Odometría/IMU: no definida en el diseño; mientras tanto el giro
      sale del flujo óptico (VisualYawEstimator) y la traslación se asume
      0, igual que en la laptop. TODO en read_motion().
    - Voz (TTS): las alertas y eventos se imprimen; conectar aquí el
      módulo de voz del equipo (pyttsx3, cola priorizada). TODO en say().

Salida por ciclo: un NavigationFrame (navigation_feed.py) — exactamente
lo que consumirá el módulo de navegación DRL.

Uso:
    python main_pi.py --map-dir maps/casa
    python main_pi.py --hazards --hazard-every 10   # peligros de piso
    Ctrl+C -> guarda mapa + landmarks y termina.
"""
import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from interfaces import MotionEstimate
from vision_bridge import fuse_observations
from navigation_feed import build_navigation_frame, safety_alerts


# ----------------------------------------------------------------------
# Ganchos de hardware pendiente
# ----------------------------------------------------------------------
class Mb1232Array:
    """Par de sensores MB1232 I2CXL-MaxSonar (izquierda / derecha).
    TODO: implementar con smbus2 cuando el hardware exista:
        - write 0x51 al registro de comando -> inicia medición
        - esperar ~100 ms -> leer 2 bytes = rango en cm
    Las direcciones y el montaje lateral vienen del diseño electrónico
    (TT_2026_B045 §7.6). Mientras no exista, read() devuelve [] y el
    sistema opera igual que en la laptop (solo cámara)."""

    LEFT_BEARING = math.radians(90)
    RIGHT_BEARING = math.radians(-90)
    MAX_RANGE_M = 7.65

    def __init__(self, addr_left=0x20, addr_right=0x21, bus=1):
        self.addr_left = addr_left
        self.addr_right = addr_right
        self.available = False   # TODO: detectar el bus I2C real

    def read(self):
        """-> list[UltrasonicReading]. TODO hardware."""
        return []


def read_motion(yaw_estimator, frame, dt):
    """Estimación de movimiento del ciclo. TODO: sustituir/fusionar con
    la odometría real cuando el equipo defina su fuente física (el
    diseño no especifica IMU). Por ahora: giro por flujo óptico,
    traslación 0 — el EKF corrige (x, y) con los rangos a landmarks."""
    delta_rot, ok = yaw_estimator.update(frame)
    return MotionEstimate(delta_trans=0.0, delta_rot=delta_rot, dt=dt), ok


class VoiceOutput:
    """Salida auditiva. TODO: conectar el módulo de voz real del equipo
    (pyttsx3 con cola priorizada: 0 = alerta interrumpe, 1 = navegación,
    2 = confirmación). Mientras tanto imprime con prefijo [VOZ]."""

    def __init__(self):
        self._last = {}

    def say(self, msg, priority=2, repeat_after_s=4.0):
        # No repetir la misma frase en ráfaga (el TTS real hará esto
        # con su cola; aquí evita inundar la consola).
        now = time.time()
        if now - self._last.get(msg, 0.0) < repeat_after_s:
            return
        self._last[msg] = now
        print(f"[VOZ p{priority}] {msg}", flush=True)


# ----------------------------------------------------------------------
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
    args = p.parse_args()

    import cv2
    from vision_bridge import (VisionBridge, VisualYawEstimator,
                               GroundHazardDetector)

    voice = VoiceOutput()
    voice.say("Iniciando sistema de navegacion", priority=2)

    bridge = VisionBridge(model_path=args.model,
                          camera_profile="picam3wide",
                          cam_height_m=args.cam_height,
                          cam_pitch_deg=args.cam_pitch)
    yaw_est = VisualYawEstimator(hfov_deg=math.degrees(bridge.hfov))
    hazard_det = GroundHazardDetector(
        camera_profile="picam3wide", cam_height_m=args.cam_height,
        cam_pitch_deg=args.cam_pitch) if args.hazards else None
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

            motion, _ = read_motion(yaw_est, frame, dt)
            detections = bridge.process(frame)
            if hazard_det is not None and frames % args.hazard_every == 0:
                last_hazards = hazard_det.process(frame, detections)
            detections = detections + last_hazards

            us_readings = ultrasonic.read()          # [] hasta tener HW
            observations = fuse_observations(detections, us_readings)
            output = slam.step(motion, observations)
            pose = (output.x, output.y, output.theta)
            mgr.update_from_scan(pose, slam.observations_for_map(),
                                 max_range=args.max_range)
            events = mgr.update_position(pose, slam.get_landmarks())

            for ev in events:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voice.say("Zona nueva", priority=2)
                elif ev["type"] == "room_recognized":
                    voice.say(f"Estas en {ev['room']}", priority=1)
                elif ev["type"] == "door_crossed":
                    voice.say("Cruzaste una puerta", priority=1)

            # Este NavigationFrame es la entrada del módulo de navegación
            # (DRL); cuando exista, se le entrega aquí.
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
