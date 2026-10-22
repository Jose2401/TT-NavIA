import argparse
import math
import os
import queue
import threading
import time

import numpy as np

from interfaces import MotionEstimate
from vision_bridge import fuseObservations, freeSpaceRays
from navigation_feed import buildNavigationFrame, safetyAlerts
from navigation.navigator import Navigator, FORWARD_STEP, TURN_STEP
from main_lap import (drawChunkMapPanel, drawNavBar, annotateCamera,
                      describeEvents, _armarSlamYMapa, _hudLine,
                      _hstackPad)

AYUDA = """
main_nav_lap.py — navegación completa en laptop (gui): slam + chunks +
visión -> Navigator (drl) -> indicaciones una a una. como no hay voz,
los comandos se escriben en la terminal y enter.

    python main_nav_lap.py --sim                  # cuarto simulado
    python main_nav_lap.py                        # cámara real
    python main_nav_lap.py --model-nav models/nav_ppo

comandos: ve a la silla | punto 3.5 2.0 | pausa | reanuda | detener |
ruta alternativa | estado

en --sim el usuario "obedece" las indicaciones (así validamos que, una
a una, sí te llevan al destino sin chocar xd); 'o' cambia a wasd manual.
"""


def startCommandThread(q):
    def loop():
        while True:
            try:
                linea = input()
            except EOFError:
                return
            q.put(linea)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def drawNavOverlay(view, navigator, mgr):
    import cv2
    st = navigator.status
    if st.goal is None:
        return view
    lo, _hz, (ox, oy) = mgr.stitched(includeDisk=False)
    h, w = lo.shape
    vh, vw = view.shape[:2]
    escala = min(vh / h, vw / w) if h and w else 1.0

    def aPx(wx, wy):
        gx = wx / mgr.resolution - ox
        gy =wy / mgr.resolution - oy
        return int(gx * escala), int(vh - 1 - gy * escala)

    gx, gy = aPx(*st.goal)
    cv2.drawMarker(view, (gx, gy), (255, 0, 200), cv2.MARKER_STAR, 18, 2)
    label = st.goal_label or "destino"
    cv2.putText(view, label, (gx + 8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 200), 1, cv2.LINE_AA)
    return view


# imprime lo que diria la voz
def printMsgs(msgs):
    for m in msgs:
        print(f"[NAV p{m.priority}] {m.text}", flush=True)


# sim donde el usuario obedece al navegador
def runSim(args):
    if not args.headless:
        import cv2
    from sim_world import (SimulatedWorld, SimulatedUltrasonicArray,
                           simulatedVision)
    from sim_world import KeyboardUser
    if args.headless and args.max_frames is None:
        args.max_frames = 500

    world = SimulatedWorld.defaultRoom()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    usArray = SimulatedUltrasonicArray(world)
    slam, mgr = _armarSlamYMapa(args, x0=0.7, y0=0.7)
    navigator = Navigator(args.model_nav, decide_every_s=0.0,
                          repeat_s=2.5)

    colaCmd = queue.Queue()
    startCommandThread(colaCmd)
    print(AYUDA)
    print("[NAV p2] Sistema listo. Diga su destino. "
          "(ej: 've a la silla', 'punto 5 3')")

    obedecer = True
    eventsLog = []
    trayectoria = []
    output = slam.getOutput()
    navFrame = None
    paso = 0

    while args.max_frames is None or paso < args.max_frames:
        paso += 1

        try:
            while True:
                linea = colaCmd.get_nowait()
                printMsgs(navigator.handle_command(linea, navFrame))
        except queue.Empty:
            pass

        forward = rot = 0.0
        if args.headless:
            key = 255
            time.sleep(0.0)
        else:
            import cv2
            key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            obedecer = not obedecer
            print(f"[modo] {'obedecer navegador' if obedecer else 'manual wasd'}")
        elif not obedecer:
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
            papu = navigator.status.last_action
            if papu == 0:
                forward = FORWARD_STEP / 2
            elif papu == 1:
                rot = TURN_STEP / 2
            elif papu == 2:
                rot = -TURN_STEP / 2
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
        describeEvents(mgr.updatePosition(pose, slam.getLandmarks()),
                       eventsLog)
        trayectoria.append((output.x, output.y))

        navFrame = buildNavigationFrame(
            slam, mgr, detections, maxRange=args.max_range,
            chunk=mgr.currentChunk, room=mgr.currentRoom,
            ultrasonic=usLecturas)

        alertas = safetyAlerts(navFrame)
        for prio, msg in alertas:
            if prio == 0:
                print(f"[ALERTA p0] {msg}")
        printMsgs(navigator.update(navFrame, mgr, alertas))

        if not args.headless:
            import cv2
            vista = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                      size=560, truePose=truePose)
            vista = drawNavOverlay(vista, navigator, mgr)
            st = navigator.status
            estado = f"sesion={st.session}"
            if st.goal_label:
                estado += f" destino={st.goal_label}"
            if st.d_goal:
                estado += f" faltan={st.d_goal:.1f}m"
            cv2.putText(vista, estado, (8, vista.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(vista, _hudLine(output, mgr, slam),
                        (8, vista.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 60, 60), 1, cv2.LINE_AA)
            barra = drawNavBar(navFrame, alertas, vista.shape[1])
            cv2.imshow("NavIA - navegacion DRL (sim)",
                       np.vstack([vista, barra]))

    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    if not args.no_save:
        mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())


# lo mismo pero con camara
def runCamera(args):
    import cv2
    from vision_bridge import VisionBridge
    from odometry import makeMotionEstimator

    print("Cargando módulo de visión (YOLOv8-seg)...")
    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile=args.camera_profile)
    motionEst = makeMotionEstimator(
        useImu=args.imu, imuBus=args.imu_bus, imuAddr=args.imu_addr,
        stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov))
    slam, mgr = _armarSlamYMapa(args)
    navigator = Navigator(args.model_nav)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video")

    colaCmd = queue.Queue()
    startCommandThread(colaCmd)
    print("[NAV p2] Sistema listo. Diga su destino.")

    eventsLog = []
    output = slam.getOutput()
    navFrame = None
    lastT = time.time()
    frames =0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                continue
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT = ahora

            try:
                while True:
                    printMsgs(navigator.handle_command(colaCmd.get_nowait(),
                                                       navFrame))
            except queue.Empty:
                pass

            motion, odo = motionEst.update(frame, dt)
            yawOk = odo["source"] != "ninguno"
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            observaciones = fuseObservations(detections, [])
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            mgr.updateFromScan(pose, slam.observationsForMap(),
                               maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            describeEvents(mgr.updatePosition(pose, slam.getLandmarks()),
                           eventsLog)

            navFrame = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom)
            alertas = safetyAlerts(navFrame)
            for prio, msg in alertas:
                if prio == 0:
                    print(f"[ALERTA p0] {msg}")
            printMsgs(navigator.update(navFrame, mgr, alertas))

            if not args.headless:
                camV = annotateCamera(frame.copy(), detections,
                                      bridge.hfov, yawOk, 1.0 / dt)
                mapV = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                         size=camV.shape[0])
                mapV = drawNavOverlay(mapV, navigator, mgr)
                arriba = _hstackPad(camV, mapV)
                barra = drawNavBar(navFrame, alertas, arriba.shape[1])
                cv2.imshow("NavIA - navegacion DRL (camara)",
                           np.vstack([arriba, barra]))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if not args.no_save:
            mgr.saveAll()
            mgr.saveLandmarks(slam.exportLandmarks())


def buildParser():
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
    p.add_argument("--imu", action="store_true",
                   help="usar IMU BNO055 como odometria")
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
