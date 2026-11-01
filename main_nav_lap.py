import argparse
import math
import os
import queue
import threading
import time

import numpy as np

from interfaces import MotionEstimate
from vision_bridge import fuseObservations, freeSpaceRays
from navigation_feed import buildNavigationFrame
from alertas import ModuloAlertas
from navigation.navigator import Navigator, FORWARD_STEP, TURN_STEP
from navigation.route_planner import PlanificadorRuta
from main_lap import (drawChunkMapPanel, drawNavBar, annotateCamera,
                      describeEvents, _armarSlamYMapa, _hudLine,
                      _hstackPad)
from interaccion import armarInteraccion

AYUDA = """
main_nav_lap.py — navegación completa en laptop (gui): slam + chunks +
visión -> Navigator (drl) -> indicaciones una a una. como no hay voz,
los comandos se escriben en la terminal y enter.

    python main_nav_lap.py --sim                  # cuarto simulado
    python main_nav_lap.py                        # cámara real
    python main_nav_lap.py --model-nav models/nav_ppo

comandos (teclado, o mic con --voz): ve a la silla | punto 3.5 2.0 |
pausa | reanuda | detener | otra ruta | donde estoy | que hay enfrente

en --sim el usuario "obedece" las indicaciones (así validamos que, una
a una, sí te llevan al destino sin chocar xd). teclas en --sim:
'o' wasd manual | 'b' suelta una caja adelante (mira la ruta
replanificarse) | 'm' persona cruzando | 'x' alertas on/off | 'q' salir.
la ruta planeada se dibuja en el mapa (cian = llega, rojo = bloqueada).
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


def drawNavOverlay(view, navigator, mgr, ruta=None):
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
        gy = wy / mgr.resolution - oy
        return int(gx * escala), int(vh - 1 - gy * escala)

    # la ruta planeada de inicio a fin (cian; roja si esta bloqueada)
    if ruta and len(ruta.get("points", [])) > 1:
        color = (255, 220, 0) if ruta["estado"] == "llega" else (0, 0, 255)
        pts = [aPx(px, py) for (px, py) in ruta["points"]]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(view, p1, p2, color, 2, cv2.LINE_AA)
        tag = {"llega": f"ruta: {ruta['pasos']} pasos",
               "bloqueada": "RUTA BLOQUEADA: buscando paso",
               "larga": "ruta larga..."}[ruta["estado"]]
        cv2.putText(view, tag, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)

    gx, gy = aPx(*st.goal)
    cv2.drawMarker(view, (gx, gy), (255, 0, 200), cv2.MARKER_STAR, 18, 2)
    label = st.goal_label or "destino"
    cv2.putText(view, label, (gx + 8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 200), 1, cv2.LINE_AA)
    return view


def drawRutaViva(mgr, slam, navigator, ruta, output, navFrame=None,
                 sizePx=860, ventanaM=10.0):
    """ventana grande con zoom alrededor del usuario: mapa local, ruta
    gruesa, landmarks con letra legible, moviles y HUD. el mapa global
    chiquito se queda en su panel de siempre; esta es pa VER la ruta."""
    import cv2
    cm = mgr.localCostmap((output.x, output.y), sizeM=ventanaM)
    lado = cm.shape[1]
    res = mgr.resolution
    half = lado // 2

    # fondo: blanco libre, gris desconocido, negro ocupado, rojo peligro
    occ = cm[0]
    img = np.full((lado, lado, 3), 190, np.uint8)
    img[occ < 0.35] = (255, 255, 255)
    img[occ > 0.65] = (30, 30, 30)
    img[cm[1] > 0.5] = (0, 0, 220)
    esc = sizePx / float(lado)
    vista = cv2.resize(img, (sizePx, sizePx),
                       interpolation=cv2.INTER_NEAREST)
    vista = cv2.flip(vista, 0)

    def aPx(wx, wy):
        cx = (wx - output.x) / res + half
        cy = (wy - output.y) / res + half
        return int(cx * esc), int(sizePx - 1 - cy * esc)

    # ruta planeada, gruesota
    if ruta and len(ruta.get("points", [])) > 1:
        color = (255, 210, 0) if ruta["estado"] == "llega" else (0, 0, 255)
        pts = [aPx(px, py) for (px, py) in ruta["points"]]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(vista, p1, p2, color, 4, cv2.LINE_AA)
        for p in pts[::3]:
            cv2.circle(vista, p, 5, color, -1)

    # landmarks con letra que si se lea (contorno blanco)
    mitad = ventanaM / 2.0 + 0.5
    for lm in slam.getLandmarks():
        if abs(lm["x"] - output.x) > mitad or abs(lm["y"] - output.y) > mitad:
            continue
        px, py = aPx(lm["x"], lm["y"])
        cv2.drawMarker(vista, (px, py), (0, 140, 255),
                       cv2.MARKER_TILTED_CROSS, 16, 3)
        if lm["label"] not in ("eco_us", "obstaculo"):
            for grosor, col in ((4, (255, 255, 255)), (1, (0, 90, 200))):
                cv2.putText(vista, lm["label"], (px + 10, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, grosor,
                            cv2.LINE_AA)

    # detecciones del ciclo: moviles en magenta, peligros en rojo
    if navFrame is not None:
        for d in navFrame.detections:
            if d.range_est is None:
                continue
            ang = output.theta + d.bearing
            px, py = aPx(output.x + d.range_est * math.cos(ang),
                         output.y + d.range_est * math.sin(ang))
            if d.moving:
                cv2.circle(vista, (px, py), 12, (255, 0, 255), 3)
                cv2.putText(vista, d.label, (px + 14, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),
                            2, cv2.LINE_AA)
            elif d.hazard:
                cv2.circle(vista, (px, py), 10, (0, 0, 255), 3)

    # destino
    st = navigator.status
    if st.goal is not None:
        gx, gy = aPx(*st.goal)
        cv2.drawMarker(vista, (gx, gy), (255, 0, 200), cv2.MARKER_STAR,
                       30, 3)
        cv2.putText(vista, st.goal_label or "destino", (gx + 16, gy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2,
                    cv2.LINE_AA)

    # usuario al centro con su flecha de hacia donde ve
    cx, cy = aPx(output.x, output.y)
    cv2.circle(vista, (cx, cy), 11, (0, 0, 255), -1)
    cv2.circle(vista, (cx, cy), 13, (255, 255, 255), 2)
    cv2.arrowedLine(vista, (cx, cy),
                    (int(cx + 42 * math.cos(output.theta)),
                     int(cy - 42 * math.sin(output.theta))),
                    (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.35)

    # hud grande
    lineas = [f"pos ({output.x:+.1f}, {output.y:+.1f})  "
              f"mira {math.degrees(output.theta):+.0f} grados  "
              f"conf {output.confidence:.2f}"]
    if st.session == "activo" and st.d_goal is not None:
        l2 = f"-> {st.goal_label or 'destino'}: faltan {st.d_goal:.1f} m"
        if ruta:
            l2 += {"llega": f"  (ruta: {ruta['pasos']} pasos)",
                   "bloqueada": "  RUTA BLOQUEADA",
                   "larga": "  (ruta larga)"}[ruta["estado"]]
        lineas.append(l2)
    else:
        lineas.append("sin ruta activa: di/escribe a donde vamos")
    for i, txt in enumerate(lineas):
        cv2.putText(vista, txt, (14, 32 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(vista, txt, (14, 32 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 160, 255), 2,
                    cv2.LINE_AA)
    return vista


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
    inter, voz, oido = armarInteraccion(
        navigator, usarVoz=args.voz, usarTts=args.tts,
        voskModel=args.vosk_model, micDevice=args.mic_device, echo=False)
    modAlertas = ModuloAlertas(
        hablarFn=lambda t: voz.decir(t, prioridad=0),
        interrumpirFn=voz._interrumpir)
    voz.esperarSi = modAlertas.sonando
    planner = PlanificadorRuta(navigator.policy)

    colaCmd = queue.Queue()
    startCommandThread(colaCmd)
    if oido is not None:
        oido.start(colaCmd, muteFn=lambda: voz.hablando)
    print(AYUDA)
    print("[NAV p2] Sistema listo. Diga su destino. "
          "(ej: 've a la silla', 'punto 5 3')")

    obedecer = True
    demoPersona = None
    eventsLog = []
    trayectoria = []
    output = slam.getOutput()
    navFrame = None
    paso = 0

    while args.max_frames is None or paso < args.max_frames:
        paso += 1

        apagar = False
        try:
            while True:
                linea = colaCmd.get_nowait()
                msgs = inter.manejarTexto(linea, navFrame)
                printMsgs(msgs)
                voz.decirMsgs(msgs)
                apagar = apagar or any(m.kind == "despedida" for m in msgs)
        except queue.Empty:
            pass
        if apagar:
            break

        forward = rot = 0.0
        if args.headless:
            key = 255
            time.sleep(0.0)
        else:
            import cv2
            key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('x'):
            estado = "activadas" if modAlertas.alternar() else "desactivadas"
            print(f"[alertas] {estado}")
        elif key == ord('b'):
            # demo: caja nueva 1.2 m adelante -> la ruta se replanifica
            ux, uy, uth = user.truePose
            world.obstacles.append((ux + 1.2 * math.cos(uth),
                                    uy + 1.2 * math.sin(uth), 0.25, "box"))
            print("[demo] caja agregada adelante, mira la ruta")
        elif key == ord('m'):
            # demo: persona que cruza enfrente durante unos segundos
            ux, uy, uth = user.truePose
            px = ux + 2.2 * math.cos(uth) - 1.6 * math.sin(uth)
            py = uy + 2.2 * math.sin(uth) + 1.6 * math.cos(uth)
            demoPersona = {"x": px, "y": py,
                           "vx": 0.022 * math.sin(uth),
                           "vy": -0.022 * math.cos(uth),
                           "vida": 180}
            print("[demo] persona cruzando")
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

        # persona demo: avanza, los sensores la ven y luego se quita
        if demoPersona is not None:
            demoPersona["x"] += demoPersona["vx"]
            demoPersona["y"] += demoPersona["vy"]
            demoPersona["vida"] -= 1
            world.obstacles.append((demoPersona["x"], demoPersona["y"],
                                    0.28, "person"))
        usLecturas = usArray.read(truePose)
        detections = simulatedVision(world, truePose)
        if demoPersona is not None:
            world.obstacles.pop()
            for d in detections:
                if d.label == "person":
                    d.moving = True
            if demoPersona["vida"] <= 0:
                demoPersona = None
        observaciones = fuseObservations(detections, usLecturas)
        output = slam.step(motion, observaciones)
        pose = (output.x, output.y, output.theta)
        mgr.updateFromScan(pose, slam.observationsForMap()
                           + freeSpaceRays(usLecturas),
                           maxRange=args.max_range)
        describeEvents(mgr.updatePosition(pose, slam.getLandmarks()),
                       eventsLog)
        if mgr.checarReloc(slam):
            voz.decir("Ubicación confirmada en el mapa conocido.", prioridad=2)
        trayectoria.append((output.x, output.y))

        navFrame = buildNavigationFrame(
            slam, mgr, detections, maxRange=args.max_range,
            chunk=mgr.currentChunk, room=mgr.currentRoom,
            ultrasonic=usLecturas)

        alertas, _recalc = modAlertas.evaluar(navFrame)
        for prio, msg in alertas:
            print(f"[ALERTA p{prio}] {msg}")
        st = navigator.status
        if st.session == "activo" and st.goal is not None:
            planner.solicitar(mgr, (output.x, output.y, output.theta),
                              st.goal, urgente=_recalc)
        else:
            planner.limpiar()
        msgs = inter.ciclo(navFrame, mgr, alertas)
        printMsgs(msgs)
        voz.decirMsgs(msgs)

        if not args.headless:
            import cv2
            vista = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                      size=560, truePose=truePose)
            vista = drawNavOverlay(vista, navigator, mgr, planner.ruta)
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
            viva = drawRutaViva(mgr, slam, navigator, planner.ruta,
                                output, navFrame)
            cv2.imshow("NavIA - ruta en vivo", viva)

    if not args.headless:
        import cv2
        cv2.destroyAllWindows()
    if not args.no_save:
        mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())


# lo mismo pero con camara
def runCamera(args):
    import cv2
    from vision_bridge import VisionBridge, GroundHazardDetector
    from odometry import makeMotionEstimator

    print("Cargando módulo de visión (YOLOv8-seg)...")
    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile=args.camera_profile,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    # puertas (entrada/salida) + hoyos vienen del segmentador: sin esto
    # en la sala de maestros no se veia ni la entrada
    detPeligros = None
    if not args.sin_hazards:
        print("Cargando segmentación de escena (puertas y peligros)...")
        detPeligros = GroundHazardDetector(cameraProfile=args.camera_profile)
    motionEst = makeMotionEstimator(
        useImu=args.imu, imuBus=args.imu_bus, imuAddr=args.imu_addr,
        stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov))
    slam, mgr = _armarSlamYMapa(args)
    navigator = Navigator(args.model_nav)
    inter, voz, oido = armarInteraccion(
        navigator, usarVoz=args.voz, usarTts=args.tts,
        voskModel=args.vosk_model, micDevice=args.mic_device, echo=False)
    modAlertas = ModuloAlertas(
        hablarFn=lambda t: voz.decir(t, prioridad=0),
        interrumpirFn=voz._interrumpir)
    voz.esperarSi = modAlertas.sonando
    planner = PlanificadorRuta(navigator.policy)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video")

    colaCmd = queue.Queue()
    startCommandThread(colaCmd)
    if oido is not None:
        oido.start(colaCmd, muteFn=lambda: voz.hablando)
    print("[NAV p2] Sistema listo. Diga su destino.")

    eventsLog = []
    ultimosPeligros = []
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

            apagar = False
            try:
                while True:
                    msgs = inter.manejarTexto(colaCmd.get_nowait(), navFrame)
                    printMsgs(msgs)
                    voz.decirMsgs(msgs)
                    apagar = apagar or any(m.kind == "despedida" for m in msgs)
            except queue.Empty:
                pass
            if apagar:
                break

            motion, odo = motionEst.update(frame, dt)
            yawOk = odo["source"] != "ninguno"
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
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
            describeEvents(mgr.updatePosition(pose, slam.getLandmarks()),
                           eventsLog)
            if mgr.checarReloc(slam):
                voz.decir("Ubicación confirmada en el mapa conocido.", prioridad=2)

            navFrame = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom)
            alertas, _recalc = modAlertas.evaluar(navFrame)
            for prio, msg in alertas:
                print(f"[ALERTA p{prio}] {msg}")
            st = navigator.status
            if st.session == "activo" and st.goal is not None:
                planner.solicitar(mgr, (output.x, output.y, output.theta),
                                  st.goal, urgente=_recalc)
            else:
                planner.limpiar()
            msgs = inter.ciclo(navFrame, mgr, alertas)
            printMsgs(msgs)
            voz.decirMsgs(msgs)

            if not args.headless:
                camV = annotateCamera(frame.copy(), detections,
                                      bridge.hfov, yawOk, 1.0 / dt)
                mapV = drawChunkMapPanel(mgr, slam, output, eventsLog,
                                         size=camV.shape[0])
                mapV = drawNavOverlay(mapV, navigator, mgr, planner.ruta)
                arriba = _hstackPad(camV, mapV)
                barra = drawNavBar(navFrame, alertas, arriba.shape[1])
                cv2.imshow("NavIA - navegacion DRL (camara)",
                           np.vstack([arriba, barra]))
                viva = drawRutaViva(mgr, slam, navigator, planner.ruta,
                                    output, navFrame)
                cv2.imshow("NavIA - ruta en vivo", viva)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('x'):
                    estado = ("activadas" if modAlertas.alternar()
                              else "desactivadas")
                    print(f"[alertas] {estado}")
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
    p.add_argument("--sin-hazards", action="store_true",
                   help="apagar segmentacion (puertas/hoyos) en camara")
    p.add_argument("--hazard-every", type=int, default=4)
    p.add_argument("--sin-extra", action="store_true",
                   help="apagar el detector extra (puertas/escaleras, oiv7)")
    p.add_argument("--extra-every", type=int, default=5)
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
    p.add_argument("--voz", action="store_true",
                   help="comandos por micrófono (vosk)")
    p.add_argument("--tts", action="store_true",
                   help="respuestas habladas (pyttsx3)")
    p.add_argument("--vosk-model", default="models/vosk-model-small-es-0.42")
    p.add_argument("--mic-device", type=int, default=None)
    return p


if __name__ == "__main__":
    args = buildParser().parse_args()
    if args.sim:
        runSim(args)
    else:
        runCamera(args)
