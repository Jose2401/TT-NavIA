import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuseObservations
from navigation_feed import buildNavigationFrame, safetyAlerts


class Mb1232Array:
    LEFT_BEARING = math.radians(90)
    RIGHT_BEARING = math.radians(-90)
    MAX_RANGE_M = 7.65

    def __init__(self, addrLeft=0x20, addrRight=0x21, bus=1):
        self.addrLeft = addrLeft
        self.addrRight= addrRight
        self.available = False

    # TODO hardware, por ahora nada
    def read(self):
        return []


class VoiceOutput:
    def __init__(self):
        self._ultimo = {}

    # no repite la misma frase en rafaga
    def say(self, msg, priority=2, repeatAfterS=4.0):
        ahora = time.time()
        if ahora - self._ultimo.get(msg, 0.0) < repeatAfterS:
            return
        self._ultimo[msg] = ahora
        print(f"[VOZ p{priority}] {msg}", flush=True)


# loop headless de la pi
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
    from odometry import makeMotionEstimator

    voz = VoiceOutput()
    voz.say("Iniciando sistema de navegacion", priority=2)

    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile="picam3wide",
                          camHeightM=args.cam_height,
                          camPitchDeg=args.cam_pitch)
    detPeligros = GroundHazardDetector(
        cameraProfile="picam3wide", camHeightM=args.cam_height,
        camPitchDeg=args.cam_pitch) if args.hazards else None
    motionEst = makeMotionEstimator(
        useImu=not args.no_imu, imuBus=args.imu_bus,
        imuAddr=args.imu_addr, stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov),
        say=lambda m: voz.say(m, priority=2))
    ultrasonico = Mb1232Array()

    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        voz.say("Mapa conocido cargado", priority=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    lastT = time.time()
    lastSave= lastT
    ultimosPeligros = []
    frames = 0
    nav = None

    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT =ahora

            motion, odo = motionEst.update(frame, dt)
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
                if detPeligros is not None:
                    detPeligros.camPitchDeg = odo["pitch_deg"]
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and frames % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros

            usLecturas = ultrasonico.read()
            observaciones = fuseObservations(detections, usLecturas)
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            mgr.updateFromScan(pose, slam.observationsForMap(),
                               maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            eventos = mgr.updatePosition(pose, slam.getLandmarks())

            for ev in eventos:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voz.say("Zona nueva", priority=2)
                elif ev["type"] == "room_recognized":
                    voz.say(f"Estas en {ev['room']}", priority=1)
                elif ev["type"] == "door_crossed":
                    voz.say("Cruzaste una puerta", priority=1)

            nav = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom,
                ultrasonic=usLecturas)

            for prio, msg in safetyAlerts(nav):
                voz.say(msg, priority=prio,
                        repeatAfterS=2.0 if prio == 0 else 5.0)

            if ahora - lastSave > args.save_every:
                mgr.saveAll()
                mgr.saveLandmarks(slam.exportLandmarks())
                lastSave = ahora
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        n = mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        print(f"\n[mapa] {n} chunks + landmarks guardados en {args.map_dir}")
        voz.say("Sistema detenido", priority=2)


if __name__ == "__main__":
    main()
