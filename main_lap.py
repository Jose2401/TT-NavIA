import argparse
import math
import os
import time

import numpy as np

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from occupancy_grid import OCC_THRESHOLD, HAZARD_MIN_HITS
from interfaces import MotionEstimate
from vision_bridge import fuseObservations, freeSpaceRays
from navigation_feed import buildNavigationFrame, safetyAlerts


# panel del mapa global con chunks, landmarks y elipse
def drawChunkMapPanel(mgr, slam, output, eventsLog, size=520,
                      truePose=None):
    import cv2
    lo, hz, (ox, oy) = mgr.stitched(includeDisk=False)
    occProb = 1.0 / (1.0 + np.exp(-lo))
    img = (255 * (1 - occProb)).astype(np.uint8)
    vista = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vista[hz >= HAZARD_MIN_HITS] = (0, 0, 220)

    h, w = vista.shape[:2]
    escala = size / float(max(h, w))
    vista = cv2.resize(vista, (int(w * escala), int(h * escala)),
                       interpolation=cv2.INTER_NEAREST)
    vista = cv2.flip(vista, 0)
    vh, vw = vista.shape[:2]

    def aPx(wx, wy):
        gx = wx / mgr.resolution - ox
        gy= wy / mgr.resolution - oy
        return int(gx * escala), int(vh - 1 - gy * escala)

    for (ci, cj) in mgr.knownChunkIndices():
        x0, y0 = aPx(ci * mgr.chunkSizeM, cj * mgr.chunkSizeM)
        x1, y1 = aPx((ci + 1) * mgr.chunkSizeM,
                     (cj + 1) * mgr.chunkSizeM)
        cv2.rectangle(vista, (x0, y0), (x1, y1), (180, 100, 180), 1)
        chunk = mgr.chunks.get((ci, cj))
        tag = f"{ci},{cj}"
        if chunk is not None and chunk.room:
            tag += f" {chunk.room}"
        cv2.putText(vista, tag, (min(x0, x1) + 3, max(y0, y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 100, 180), 1,
                    cv2.LINE_AA)

    for lm in slam.getLandmarks():
        px, py = aPx(lm["x"], lm["y"])
        if 0 <= px < vw and 0 <= py < vh:
            cv2.drawMarker(vista, (px, py), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 8, 1)
            if lm["label"] not in ("eco_us", "obstaculo"):
                cv2.putText(vista, lm["label"], (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255),
                            1, cv2.LINE_AA)

    if truePose is not None:
        gx, gy = aPx(truePose[0], truePose[1])
        cv2.circle(vista, (gx, gy), 5, (0, 180, 0), -1)

    P = np.array(output.covariance)[:2, :2]
    vals, vecs = np.linalg.eigh(P)
    vals =np.maximum(vals, 0.0)
    pxPorM = escala / mgr.resolution
    ejes = (max(2, int(2 * math.sqrt(vals[1]) * pxPorM)),
            max(2, int(2 * math.sqrt(vals[0]) * pxPorM)))
    ang = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    px, py = aPx(output.x, output.y)
    cv2.ellipse(vista, (px, py), ejes, ang, 0, 360, (0, 0, 200), 1,
                cv2.LINE_AA)
    cv2.circle(vista, (px, py), 6, (0, 0, 255), -1)
    cv2.line(vista, (px, py),
             (int(px + 22 * math.cos(output.theta)),
              int(py - 22 * math.sin(output.theta))), (0, 0, 255), 2)

    for i, txt in enumerate(eventsLog[-3:]):
        cv2.putText(vista, txt, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 80, 200), 1, cv2.LINE_AA)
    return vista


def annotateCamera(frame, detections, hfov, yawOk, fps):
    import cv2
    h, w = frame.shape[:2]
    for det in detections:
        col = int(w / 2.0 - (det.bearing / hfov) * w)
        if det.hazard:
            color = (255, 0, 255)
        elif det.moving:
            color= (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.line(frame, (col, int(h * 0.30)), (col, int(h * 0.70)), color, 2)
        tag = det.label
        if det.estado:
            tag += f" ({det.estado})"
        if det.range_est is not None:
            tag += f" {det.range_est:.1f}m"
        if det.moving:
            tag += " [movil]"
        if det.hazard:
            tag += " [PELIGRO PISO]"
        cv2.putText(frame, tag, (max(0, col - 60), int(h * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    estado = "giro visual: OK" if yawOk else "giro visual: SIN TEXTURA"
    cv2.putText(frame, f"{estado} | {fps:.1f} fps | q: salir y guardar",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def drawNavBar(frameNav, alerts, width, height=64):
    import cv2
    barra = np.zeros((height, width, 3), dtype=np.uint8)
    colorRiesgo = {0: (0, 200, 0), 1: (0, 200, 200),
                   2: (0, 140, 255), 3: (0, 0, 255)}

    def fmt(d):
        return "libre" if d >= 3.99 else f"{d:.2f}m"

    items = (("IZQ", frameNav.d_left, frameNav.risk_left),
             ("FRENTE", frameNav.d_front, frameNav.risk_front),
             ("DER", frameNav.d_right, frameNav.risk_right))
    seg =width // 3
    for i, (nombre, d, r) in enumerate(items):
        cv2.putText(barra, f"{nombre}: {fmt(d)} r={r}",
                    (i * seg + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    colorRiesgo[r], 2, cv2.LINE_AA)
    if alerts:
        prio, msg = sorted(alerts)[0]
        color = (0, 0, 255) if prio == 0 else (0, 165, 255)
        cv2.putText(barra, msg, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    else:
        cv2.putText(barra, "sin alertas", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1,
                    cv2.LINE_AA)
    return barra


def describeEvents(events, log):
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
            txt =str(ev)
        print(f"[evento] {txt}")
        log.append(txt)


# loop con camara real
def runCamera(args):
    import cv2
    from vision_bridge import VisionBridge, GroundHazardDetector
    from odometry import makeMotionEstimator

    print("Cargando módulo de visión (YOLOv8-seg de paquete vision/)...")
    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile=args.camera_profile,
                          camHeightM=args.cam_height,
                          camPitchDeg=args.cam_pitch,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    motionEst = makeMotionEstimator(
        useImu=args.imu, imuBus=args.imu_bus, imuAddr=args.imu_addr,
        stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov))
    detPeligros = None
    if args.hazards:
        print("Cargando segmentación de escena (SegFormer ADE20K) para "
              "peligros de piso...")
        detPeligros = GroundHazardDetector(
            cameraProfile=args.camera_profile,
            camHeightM=args.cam_height, camPitchDeg=args.cam_pitch)

    fuente = args.video if args.video else args.camera
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {fuente}")

    slam, mgr = _armarSlamYMapa(args)
    trayectoria = []
    output = slam.getOutput()
    eventsLog = []
    ultimosPeligros = []
    lastT = time.time()
    fps = 0.0
    frames= 0

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
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT =ahora
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            motion, odo = motionEst.update(frame, dt)
            yawOk = odo["source"] != "ninguno"
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
                if detPeligros is not None:
                    detPeligros.camPitchDeg = odo["pitch_deg"]

            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and frames % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros

            observaciones = fuseObservations(detections, [])
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            mgr.updateFromScan(pose, slam.observationsForMap(),
                               maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            eventos = mgr.updatePosition(pose, slam.getLandmarks())
            describeEvents(eventos, eventsLog)
            if mgr.checarReloc(slam):
                print("[mapa] re-localizacion confirmada en el mapa conocido")
            trayectoria.append((output.x, output.y))

            nav = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom)
            alertas = safetyAlerts(nav)
            for prio, msg in alertas:
                if prio == 0:
                    print(f"[ALERTA] {msg}")

            if not args.headless:
                camV = annotateCamera(frame.copy(), detections,
                                      bridge.hfov, yawOk, fps)
                mapV = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                         size=camV.shape[0])
                hud = _hudLine(output, mgr, slam)
                cv2.putText(camV, hud, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60),
                            2, cv2.LINE_AA)
                arriba = _hstackPad(camV, mapV)
                barra = drawNavBar(nav, alertas, arriba.shape[1])
                cv2.imshow("NavIA - laptop: vision | mapa por chunks",
                           np.vstack([arriba, barra]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    n = mgr.saveAll()
                    mgr.saveLandmarks(slam.exportLandmarks())
                    print(f"[mapa] guardados {n} chunks")
            elif frames % 30 == 0:
                print(f"[f={frames:04d}] {_hudLine(output, mgr, slam)} "
                      f"d=({nav.d_front:.1f},{nav.d_left:.1f},"
                      f"{nav.d_right:.1f})")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    _terminar(args, mgr, slam, output, trayectoria)


# loop con el cuarto simulado
def runSim(args):
    from sim_world import (SimulatedWorld, SimulatedMotion,
                           SimulatedUltrasonicArray, simulatedVision)
    from sim_world import KeyboardUser

    if not args.headless:
        try:
            import cv2
        except ImportError:
            print("[aviso] OpenCV no disponible: modo headless.")
            args.headless = True
    if args.headless and args.max_frames is None:
        args.max_frames = 400

    world = SimulatedWorld.defaultRoom()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    piloto = SimulatedMotion(start=(0.7, 0.7, 0.0))
    usArray = SimulatedUltrasonicArray(world)

    slam, mgr = _armarSlamYMapa(args, x0=0.7, y0=0.7)
    autopiloto = True
    trayectoria, gtTraj = [], []
    eventsLog = []
    output = slam.getOutput()
    paso = 0

    print("Modo simulado con chunks de "
          f"{mgr.chunkSizeM} m. wasd para moverte, p: autopiloto, "
          "q: salir y guardar.")
    while args.max_frames is None or paso < args.max_frames:
        paso += 1
        if autopiloto:
            piloto.truePose = list(user.truePose)
            motion = piloto.get()
            user.truePose = list(piloto.truePose)
        else:
            import cv2
            forward = rot = 0.0
            key = cv2.waitKey(40) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                autopiloto = True
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

        truePose = tuple(user.truePose)
        usLecturas = usArray.read(truePose)
        detections = simulatedVision(world, truePose)
        observaciones = fuseObservations(detections, usLecturas)

        output = slam.step(motion, observaciones)
        pose = (output.x, output.y, output.theta)
        mgr.updateFromScan(pose, slam.observationsForMap()
                           + freeSpaceRays(usLecturas),
                           maxRange=args.max_range)
        eventos = mgr.updatePosition(pose, slam.getLandmarks())
        describeEvents(eventos, eventsLog)
        if mgr.checarReloc(slam):
            print("[mapa] re-localizacion confirmada en el mapa conocido")
        trayectoria.append((output.x, output.y))
        gtTraj.append((truePose[0], truePose[1]))

        nav = buildNavigationFrame(slam, mgr, detections,
                                   maxRange=args.max_range,
                                   chunk=mgr.currentChunk,
                                   room=mgr.currentRoom,
                                   ultrasonic=usLecturas)
        alertas = safetyAlerts(nav)

        if not args.headless:
            import cv2
            mapV = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                     size=560, truePose=truePose)
            cv2.putText(mapV, _hudLine(output, mgr, slam), (8, mapV.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 1,
                        cv2.LINE_AA)
            barra = drawNavBar(nav, alertas, mapV.shape[1])
            cv2.imshow("NavIA - sim: mapa por chunks", np.vstack([mapV, barra]))
            if autopiloto:
                key = cv2.waitKey(40) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    autopiloto = False
        elif paso % 50 == 0:
            err = math.hypot(output.x - truePose[0],
                             output.y - truePose[1])
            print(f"[t={paso:04d}] {_hudLine(output, mgr, slam)} "
                  f"err={err:.2f}m")

    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    _terminar(args, mgr, slam, output, trayectoria, gtTraj)


def _armarSlamYMapa(args, x0=0.0, y0=0.0):
    mapDir = None if args.no_save else args.map_dir
    if mapDir and args.fresh and os.path.isdir(mapDir):
        import shutil
        shutil.rmtree(mapDir)
        print(f"[mapa] {mapDir} borrado (--fresh)")
    mgr = ChunkedMapManager(mapDir=mapDir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(x0, y0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        mgr.marcarRelocPendiente()
        print(f"[mapa] {len(guardados)} landmarks restaurados de {mapDir} "
              "(re-localización activa)")
    if mgr.knownChunkIndices():
        print(f"[mapa] mapa conocido: {len(mgr.knownChunkIndices())} "
              f"chunks en {mapDir}")
    return slam, mgr


def _hudLine(output, mgr, slam):
    room = f" cuarto={mgr.currentRoom}" if mgr.currentRoom else ""
    return (f"x={output.x:+.2f} y={output.y:+.2f} "
            f"th={math.degrees(output.theta):+.0f}deg "
            f"conf={output.confidence:.2f} "
            f"chunk={mgr.currentChunk}{room} "
            f"lm={len(slam.confirmedLandmarks())}")


def _hstackPad(a, b):
    import cv2
    h = max(a.shape[0], b.shape[0])

    def pad(img):
        if img.shape[0] == h:
            return img
        out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
        out[:img.shape[0]] = img
        return out
    return np.hstack([pad(a), pad(b)])


def _terminar(args, mgr, slam, output, trayectoria, gtTraj=None):
    if not args.no_save:
        n = mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        print(f"[mapa] {n} chunks + landmarks guardados en {mgr.mapDir}")
    path = mgr.renderCroquis(
        pose=(output.x, output.y, output.theta),
        trajectory=trayectoria, gtTrajectory=gtTraj,
        landmarks=slam.confirmedLandmarks(),
        path=args.croquis)
    print(f"Croquis global guardado en: {path}")
    print(f"Pose final: x={output.x:.2f} y={output.y:.2f} "
          f"theta={math.degrees(output.theta):.1f} deg "
          f"(confianza={output.confidence:.2f})")


def buildParser():
    p = argparse.ArgumentParser(
        description="Main de prueba en laptop (GUI) del pipeline "
                    "SLAM + mapeo por chunks + visión")
    p.add_argument("--sim", action="store_true",
                   help="cuarto simulado en vez de cámara")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--sin-extra", action="store_true",
                   help="apagar el detector extra (puertas/escaleras, oiv7)")
    p.add_argument("--extra-every", type=int, default=5)
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
    p.add_argument("--imu", action="store_true",
                   help="usar IMU BNO055 por I2C como odometría "
                        "(giro + pasos); sin ella, flujo óptico")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x28)
    p.add_argument("--imu-bus", type=int, default=1)
    p.add_argument("--step-length", type=float, default=0.68)
    return p


if __name__ == "__main__":
    args = buildParser().parse_args()
    if args.sim:
        runSim(args)
    else:
        runCamera(args)
