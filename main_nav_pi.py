import argparse
import math
import os
import queue
import signal
import threading
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuseObservations, validarConUltrasonido
from navigation_feed import buildNavigationFrame
from alertas import ModuloAlertas, BotonPullUp, conectarUltrasonido
from navigation.navigator import Navigator
from main_pi import (abrirGafas, engancharEcos, Cronometro, _aprenderPaso,
                     _alDisco)
from odometry import makeMotionEstimator
from interaccion import armarInteraccion, conectarOido


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
    p.add_argument("--camera", default="0",
                   help="índice de cámara (la csi de las gafas) o un video")
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--sin-extra", action="store_true",
                   help="apagar el detector extra (puertas/escaleras, oiv7)")
    p.add_argument("--extra-every", type=int, default=3,
                   help="cada cuántos cuadros corre el detector de "
                        "puertas/escaleras (con 8 una puerta nunca llegaba "
                        "a quedar en el mapa)")
    p.add_argument("--grabar", action="store_true",
                   help="grabar la corrida (cuadros, sensores, lo que oyó "
                        "y dijo) en sesiones/nav_FECHA pa repasarla después")
    p.add_argument("--model-nav", default="models/nav_ppo_policy.npz",
                   help="política de navegación (el .npz corre con numpy)")
    p.add_argument("--hazards", action="store_true")
    p.add_argument("--hazard-every", type=int, default=10)
    p.add_argument("--cam-height", type=float, default=None,
                   help="altura de la cámara al piso (m); por defecto la "
                        "de montaje.py")
    p.add_argument("--map-dir", default=os.path.join("maps", "default"))
    p.add_argument("--fresh", action="store_true",
                   help="borrar el mapa guardado y empezar de cero (lo va "
                        "armando y reconociendo en esta misma corrida)")
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-every", type=float, default=30.0)
    p.add_argument("--no-imu", action="store_true",
                   help="no usar la IMU BNO055 (solo flujo óptico)")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x28)
    p.add_argument("--imu-bus", type=int, default=1)
    p.add_argument("--step-length", type=float, default=None)
    p.add_argument("--puerto", default=None,
                   help="puerto serie del RP2040 de las gafas (auto)")
    p.add_argument("--sin-gafas", action="store_true",
                   help="no buscar el RP2040 (sensores por i2c o ninguno)")
    p.add_argument("--no-voz", action="store_true",
                   help="sin micrófono/vosk (comandos por stdin)")
    p.add_argument("--no-tts", action="store_true",
                   help="sin voz de salida (solo consola)")
    p.add_argument("--voz", default="auto", choices=("auto", "piper", "edge"),
                   help="motor de voz: piper (sin internet) o edge "
                        "(la del equipo, ocupa internet)")
    p.add_argument("--escucha", default="auto",
                   choices=("auto", "boton", "continua"),
                   help="boton: solo escucha tras un click del botón; "
                        "continua: siempre. auto = boton si hay botón")
    p.add_argument("--sin-boton", action="store_true")
    p.add_argument("--sin-pip", action="store_true",
                   help="alertas sin pitidos (las críticas se hablan)")
    p.add_argument("--umbral-alerta", type=float, default=None,
                   help="a cuántos metros de un obstáculo fijo suena la "
                        "alerta hablada (por defecto 0.25)")
    p.add_argument("--umbral-precaucion", type=float, default=None,
                   help="a cuántos metros empieza el pitido de precaución "
                        "(por defecto 0.60)")
    p.add_argument("--mic-mudo-al-hablar", action="store_true",
                   help="no escuchar mientras el sistema habla (como "
                        "antes); úsalo si el micrófono oye a la bocina")
    p.add_argument("--vosk-model", default="models/vosk-model-small-es-0.42")
    p.add_argument("--mic-device", type=int, default=None)
    p.add_argument("--estadisticas", type=float, default=0.0,
                   help="imprimir cuadros/s y tiempos cada N segundos")
    args = p.parse_args()

    from camara import abrirCamara
    from vision_bridge import (VisionBridge, GroundHazardDetector,
                               freeSpaceRays)
    import montaje
    import audio_pi

    # si lo paran con systemd (SIGTERM) tambien guarda el mapa
    def _sigterm(_s, _f):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except ValueError:
        pass

    # ---- hardware primero: si falta la camara ni pa que seguir ----
    cap, perfil = abrirCamara(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara "
                         f"({getattr(cap, 'razon', '')})")
    sensores, ultrasonico = abrirGafas(args.puerto, usar=not args.sin_gafas)

    # un click = escuchar / dejar de escuchar; doble = alertas on/off
    oido = None

    def _clickEscucha():
        if oido is None or not oido.hayMic:
            voz.decir("El micrófono no está disponible.", prioridad=2,
                      cadaS=3.0)
            return
        if not oido.escuchando:
            voz.callar()        # te cede la palabra: se calla y escucha
        oido.alternar()

    def _toggleAlertas():
        est = "activadas" if modAlertas.alternar() else "desactivadas"
        boton.parpadear()
        voz.decir(f"Alertas {est}.", prioridad=1)

    boton = BotonPullUp(alCambiar=_toggleAlertas, alClick=_clickEscucha,
                        usarGpio=not args.sin_boton)
    if not args.sin_boton:
        print("[BOTON] " + ("listo (1 click = escuchar, 2 = alertas)"
                            if boton.disponible
                            else f"NO disponible ({boton.razon})"))
    modoEscucha = args.escucha
    if modoEscucha == "auto":
        modoEscucha = "boton" if boton.disponible else "continua"
    modoEscucha = "continuo" if modoEscucha == "continua" else modoEscucha

    # ---- interaccion ----
    navigator = Navigator(args.model_nav)
    inter, voz, oido = armarInteraccion(
        navigator, usarVoz=not args.no_voz, usarTts=not args.no_tts,
        voskModel=args.vosk_model, micDevice=args.mic_device,
        motorVoz=args.voz, modoEscucha=modoEscucha)
    modAlertas = ModuloAlertas(
        hablarFn=voz.alerta, interrumpirFn=voz._interrumpir,
        conPip=not args.sin_pip, criticaM=args.umbral_alerta,
        precaucionM=args.umbral_precaucion)
    print(f"[ALERTAS] voz a {modAlertas.criticaM:.2f} m o menos, pitido "
          f"de precaución desde {modAlertas.precaucionM:.2f} m")
    voz.esperarSi = modAlertas.sonando
    inter.alAlertas = modAlertas.activar
    if hasattr(voz.engine, "precalentar"):
        # las frases de alerta critica ya sintetizadas desde el arranque
        d = inter.dialogo
        voz.engine.precalentar(
            [d.obstacle_warning(t, lado, None, 3)
             for _ in range(3) for t in ("pared", "obstaculo", "person")
             for lado in ("frente", "izquierda", "derecha")]
            + ["Alertas activadas.", "Alertas desactivadas."])

    if oido is not None:
        def _cambioEscucha(encendido):
            print("[ESCUCHA] " + ("micrófono abierto" if encendido
                                  else "micrófono cerrado"), flush=True)
            boton.led(encendido)
            audio_pi.tonito("escucha_on" if encendido else "escucha_off")
        oido.alCambiar = _cambioEscucha
    # sin saludo ni presentacion (pedido del equipo): nomas avisa que ya
    # cargo, pq sin monitor no hay otra forma de saberlo
    voz.decir("Listo.", prioridad=2)
    if sensores is not None and not args.sin_boton and not boton.disponible:
        # con las gafas puestas y sin boton, el micro queda siempre
        # abierto y no hay como apagar las alertas: que se sepa
        voz.decir("Atención: el botón no responde. Te escucho todo el "
                  "tiempo.", prioridad=1)

    # ---- percepcion ----
    bridge = VisionBridge(modelPath=args.model, cameraProfile=perfil,
                          camHeightM=args.cam_height or montaje.CAM_ALTURA_M,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    detPeligros = GroundHazardDetector(
        cameraProfile=perfil,
        camHeightM=args.cam_height or montaje.CAM_ALTURA_M) \
        if args.hazards else None
    motionEst = makeMotionEstimator(
        useImu=not args.no_imu, imuBus=args.imu_bus,
        imuAddr=args.imu_addr, stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov),
        say=lambda m: print(f"[ODOMETRIA] {m}"), sensores=sensores)
    if motionEst.source != "imu":
        voz.decir("Atención: no detecto la brújula de los lentes.",
                  prioridad=1)
    if not ultrasonico.available:
        voz.decir("Atención: no detecto los sensores de distancia.",
                  prioridad=1)
    # los ecos pitan desde el hilo del sensor, sin esperar a la camara
    pitchActual = [None]
    ecosEnHilo = engancharEcos(sensores, modAlertas, lambda: pitchActual[0])

    if args.fresh and os.path.isdir(args.map_dir):
        import shutil
        shutil.rmtree(args.map_dir)
        print(f"[mapa] {args.map_dir} borrado (--fresh)")
    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        mgr.marcarRelocPendiente()
        voz.decir("Mapa conocido cargado.", prioridad=2)

    grab = None
    if args.grabar:
        try:
            from grabar_sesion import GrabadorCorrida
            grab = GrabadorCorrida(sensores, meta={
                "umbrales": [modAlertas.criticaM, modAlertas.precaucionM],
                "mapa": args.map_dir})
            print(f"[GRABAR] esta corrida queda en {grab.carpeta}")
            decirOrig = voz.decir

            def decirYAnotar(texto, prioridad=2, *a, **k):
                grab.evento(f"voz{prioridad}", texto)
                return decirOrig(texto, prioridad, *a, **k)
            voz.decir = decirYAnotar
        except Exception as e:
            print(f"[GRABAR] no se pudo grabar la corrida ({e})")
            grab = None

    # el mic y el teclado meten frases a la misma cola
    colaCmd = queue.Queue()
    startCommandThread(colaCmd)
    # el micro sigue abierto mientras el sistema habla (su propia voz
    # se quita por texto) y el usuario tiene la palabra
    conectarOido(oido, voz, colaCmd, mudoAlHablar=args.mic_mudo_al_hablar)

    lastT = time.time()
    lastSave =lastT
    ultimosPeligros = []
    navFrame = None
    frames = 0
    sensoresOk = sensores is not None
    camaraOk = True
    crono = Cronometro()
    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                # si se afloja el cable de la camara a media caminata hay
                # que decirlo (los ultrasonicos siguen avisando solos)
                if camaraOk and getattr(cap, "sinImagen", lambda: False)():
                    camaraOk = False
                    voz.decir("Atención: se perdió la imagen de la cámara.",
                              prioridad=1)
                continue
            if not camaraOk:
                camaraOk = True
                voz.decir("Imagen de la cámara recuperada.", prioridad=1)
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT= ahora
            crono.inicio()

            try:
                while True:
                    dicho = colaCmd.get_nowait()
                    if grab is not None:
                        conocidas = sorted({lm.get("label") for lm in
                                            (navFrame.landmarks if navFrame
                                             else [])})
                        grab.evento("usuario", f"{dicho} | marcas conocidas: "
                                               f"{', '.join(conocidas) or 'ninguna'}")
                    msgs = inter.manejarTexto(dicho, navFrame)
                    voz.decirMsgs(msgs)
                    if any(m.kind == "despedida" for m in msgs):
                        raise KeyboardInterrupt
            except queue.Empty:
                pass

            # si se desconecta el cable de las gafas hay que decirlo: sin
            # ultrasonido el usuario se queda sin la alerta mas rapida
            if sensores is not None and sensores.available != sensoresOk:
                sensoresOk = sensores.available
                voz.decir("Sensores de los lentes reconectados."
                          if sensoresOk else
                          "Atención: se desconectaron los sensores de los "
                          "lentes.", prioridad=1)

            motion, odo = motionEst.update(frame, dt)
            pitchActual[0] = odo["pitch_deg"]
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
                if detPeligros is not None:
                    detPeligros.camPitchDeg = odo["pitch_deg"]
            crono.etapa("odometria")
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and frames % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros
            crono.etapa("vision")

            # los frontales validan a la camara (y la imu quita el
            # rebote en el piso cuando agachas la cabeza)
            usLecturas = validarConUltrasonido(
                detections, ultrasonico.read(), odo["pitch_deg"])
            if not ecosEnHilo:
                conectarUltrasonido(modAlertas, usLecturas,
                                    pitchDeg=odo["pitch_deg"])
            observaciones = fuseObservations(detections, usLecturas)
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            crono.etapa("slam")
            mgr.updateFromScan(pose, slam.observationsForMap()
                               + freeSpaceRays(usLecturas, maxLibreM=2.0)
                               + (detPeligros.tomarPisoLibre() if detPeligros else []),
                           maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            eventos = mgr.updatePosition(pose, slam.getLandmarks())
            if mgr.checarReloc(slam):
                voz.decir("Ubicación confirmada en el mapa conocido.", prioridad=2)
            for ev in eventos:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voz.decir("Zona nueva.", prioridad=2, cadaS=5.0)
                elif ev["type"] == "room_recognized":
                    voz.decir(f"Está en {ev['room']}.", prioridad=1,
                              cadaS=5.0)

            navFrame = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom,
                ultrasonic=usLecturas)
            crono.etapa("mapa")

            alertas, recalc = modAlertas.evaluar(navFrame)
            for prio, msg in alertas:
                print(f"[ALERTA p{prio}] {msg}")
            if grab is not None:
                grab.cuadro(frame, getattr(cap, "tFrame", None))
                for prio, msg in alertas:
                    grab.evento(f"alerta{prio}", msg)
                st = navigator.status
                grab.evento("pose", f"{output.x:.2f} {output.y:.2f} "
                            f"{math.degrees(output.theta):.0f} pasos "
                            f"{odo['steps_total']} sesion {st.session} "
                            f"meta {st.goal_label} "
                            f"{'' if st.d_goal is None else round(st.d_goal, 1)}")
            voz.decirMsgs(inter.ciclo(navFrame, mgr, alertas,
                                      recalcular=recalc))
            crono.etapa("alertas+nav")
            crono.cuadro()
            if args.estadisticas and ahora - crono.t0 > args.estadisticas:
                print(f"[CICLO] {crono.resumen()} | pose ({output.x:.1f}, "
                      f"{output.y:.1f}, {math.degrees(output.theta):.0f}°) "
                      f"pasos {odo['steps_total']}", flush=True)
                crono.reiniciar()

            if ahora - lastSave > args.save_every:
                mgr.saveAll()
                mgr.saveLandmarks(slam.exportLandmarks())
                _alDisco()
                lastSave = ahora
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        _alDisco()
        if oido is not None:
            oido.cerrar()
        os.sync()          # todo en la tarjeta antes de despedirse
        voz.decir("Sistema detenido.", prioridad=2)
        voz.cerrar()
        if grab is not None:
            grab.cerrar(pasos=odo["steps_total"] if frames else 0)
            print(f"[GRABAR] corrida guardada en {grab.carpeta} "
                  f"({grab.n} cuadros)")
        if sensores is not None:
            _aprenderPaso(sensores)
            sensores.cerrar()
        boton.led(False)
        boton.cerrar()


if __name__ == "__main__":
    main()
    # ya se guardo todo en el finally. salida directa: asi ningun hilo de
    # las librerias nativas (onnxruntime, libcamera) aborta al cerrar y
    # systemd ve un 0 limpio (no lo vuelve a levantar tras "apagar sistema")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
