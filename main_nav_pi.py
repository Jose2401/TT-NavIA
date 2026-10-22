import argparse
import math
import os
import queue
import threading
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuseObservations
from navigation_feed import buildNavigationFrame, safetyAlerts
from navigation.navigator import Navigator
from main_pi import Mb1232Array, VoiceOutput
from odometry import makeMotionEstimator


# lee comandos de stdin en otro hilo
def startCommandThread(q):
    def loop():
        while True:
            try:
                q.put(input())
            except EOFError:
                return
    threading.Thread(target=loop, daemon=True).start()


# navegacion completa en la pi, todo por voz
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

    voz = VoiceOutput()
    voz.say("Sistema listo. Diga su destino.", priority=2)

    bridge = VisionBridge(modelPath=args.model, cameraProfile="picam3wide")
    detPeligros = GroundHazardDetector(cameraProfile="picam3wide") \
        if args.hazards else None
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
        voz.say("Mapa conocido cargado.", priority=2)

    navigator = Navigator(args.model_nav)
    colaCmd = queue.Queue()
    startCommandThread(colaCmd)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    lastT = time.time()
    lastSave =lastT
    ultimosPeligros = []
    navFrame = None
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT= ahora

            try:
                while True:
                    for m in navigator.handle_command(colaCmd.get_nowait(),
                                                      navFrame):
                        voz.say(m.text, priority=m.priority)
            except queue.Empty:
                pass

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
                    voz.say("Zona nueva.", priority=2)
                elif ev["type"] == "room_recognized":
                    voz.say(f"Está en {ev['room']}.", priority=1)

            navFrame = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom,
                ultrasonic=usLecturas)

            alertas = safetyAlerts(navFrame)
            for prio, msg in alertas:
                voz.say(msg, priority=prio,
                        repeatAfterS=2.0 if prio == 0 else 5.0)
            for m in navigator.update(navFrame, mgr, alertas):
                voz.say(m.text, priority=m.priority)

            if ahora - lastSave > args.save_every:
                mgr.saveAll()
                mgr.saveLandmarks(slam.exportLandmarks())
                lastSave = ahora
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        voz.say("Sistema detenido.", priority=2)


if __name__ == "__main__":
    main()
