"""
sesion_a_mapa.py — una grabacion de grabar_sesion.py -> mapa guardado,
pasandola por el MISMO pipeline que corre en vivo (vision + imu de las
gafas + ultrasonicos + ekf + ray casting). es el hermano de
video_a_mapa.py pero con los sensores reales: aqui la escala del mapa
sale de los pasos contados por la imu, no de una velocidad supuesta.

uso:
    python sesion_a_mapa.py sesiones/pasillo1 --map-dir maps/real/p01 --fresh
    python sesion_a_mapa.py sesiones/a sesiones/b --map-dir maps/real/p02

al final imprime el reporte de sensores (pasos, giro, ruido de los
ultrasonicos) y si el mapa ya sirve pa entrenar/evaluar la navegacion.
"""
import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import (fuseObservations, validarConUltrasonido,
                           freeSpaceRays)


def tiemposDeCuadros(carpeta):
    """[(t, ruta)] de los cuadros que SI tienen imagen. si cuadros.csv
    no existe o quedo vacio (la pi se apago antes de que se escribiera,
    paso el 2026-11-12) los tiempos se rescatan de la hora de cada jpg:
    el primero se escribio ~0.1 s despues de empezar la grabacion."""
    dirCuadros = os.path.join(carpeta, "cuadros")
    cuadros = []
    try:
        with open(os.path.join(carpeta, "cuadros.csv"), encoding="utf-8") as fh:
            next(fh)
            for renglon in fh:
                i, t = renglon.strip().split(",")
                cuadros.append((float(t),
                                os.path.join(dirCuadros, f"{int(i):06d}.jpg")))
    except (OSError, StopIteration, ValueError):
        cuadros = []
    buenos = lambda ruta: os.path.exists(ruta) and os.path.getsize(ruta) > 0
    if cuadros:
        return [(t, r) for t, r in cuadros if buenos(r)]
    rutas = sorted(os.path.join(dirCuadros, f) for f in os.listdir(dirCuadros)
                   if f.endswith(".jpg"))
    rutas = [r for r in rutas if buenos(r)]
    if not rutas:
        return []
    t0 = os.stat(rutas[0]).st_mtime - 0.1
    return [(os.stat(r).st_mtime - t0, r) for r in rutas]


def leerSesion(carpeta):
    """-> (cuadros [(t, ruta)], lineas [(t, texto)]) ordenados por t."""
    cuadros = tiemposDeCuadros(carpeta)
    lineas = []
    with open(os.path.join(carpeta, "serie.txt"), encoding="utf-8") as fh:
        for renglon in fh:
            t, _, texto = renglon.strip().partition(" ")
            try:
                lineas.append((float(t), texto))
            except ValueError:
                pass
    return cuadros, lineas


class _Repaso:
    """lo que se junta al repasar una grabacion pa poder EVALUAR el
    sistema: que detecto la vision (y cuantas veces), las alertas que
    habria dado, y el video con todo eso dibujado encima."""

    def __init__(self, carpeta, bridge, fps, conVideo):
        import collections
        from alertas import ModuloAlertas
        self.conteo = collections.Counter()        # etiqueta -> cuadros
        self.confMax = {}
        self.alertas = []                          # (t, nivel, texto)
        self.reloj = [0.0]
        self.mod = ModuloAlertas(conPip=False)
        self.mod.reloj = lambda: self.reloj[0]
        self.mod.salida = self                     # los "pips" caen aqui
        self.CRITICAL, self.WARNING = 3, 2
        self.video = None
        self._aviso = (0.0, "")
        self._crudas = {"coco": [], "extra": []}
        if conVideo:
            from sesion_a_video import VideoAnotado
            self.video = VideoAnotado(
                os.path.join(carpeta, "video_sistema.mp4"), fps=fps)
        # las cajas crudas del detector (la VisionDetection ya no las trae)
        detectar = bridge.detector.detect
        extras = bridge._detectarExtras

        def detect(frame):
            self._crudas["coco"] = detectar(frame)
            self._crudas["extra"] = []
            return self._crudas["coco"]

        def detectarExtras(frame):
            self._crudas["extra"] = extras(frame)
            return self._crudas["extra"]
        self._restaurar = (bridge, detectar, extras)
        bridge.detector.detect = detect
        bridge._detectarExtras = detectarExtras

    def pip(self, nivel, direccion, tipo, dist):
        cm = "" if dist is None else f" a {dist:.2f} m"
        self.alertas.append((self.reloj[0], nivel, f"{tipo} {direccion}{cm}"))
        self._aviso = (self.reloj[0], ("ALERTA CON VOZ: " if nivel >= 3
                                       else "pitido: ")
                       + f"{tipo} {direccion}{cm}")

    def contar(self, detections):
        for d in {d.label: d for d in detections}.values():
            self.conteo[d.label] += 1
        for d in detections:
            c = getattr(d, "confidence", 0.0) or 0.0
            self.confMax[d.label] = max(self.confMax.get(d.label, 0.0), c)

    def dibujar(self, frame, t, sens, pitch, heading, lecturas, detections,
                pose, hfov):
        if self.video is None:
            return
        import cv2
        from sesion_a_video import (dibujarSensores, dibujarCajas,
                                    dibujarAviso)
        from nombres import nombreEs
        img = cv2.resize(frame, (640, 360))
        porLabel = {}
        for d in detections:
            porLabel.setdefault(d.label, []).append(d)
        cajas = []
        for origen, color in (("coco", (90, 220, 90)), ("extra", (0, 165, 255))):
            for det in self._crudas[origen]:
                x1, y1, x2, y2 = det["bbox"]
                label = det["label"].lower()
                label = getattr(self._restaurar[0], "aliases", {}).get(label, label)
                # la distancia: la deteccion de esa clase mas cercana en rumbo
                cx = (x1 + x2) / 2.0
                rumbo = ((640 / 2.0 - cx) / 640) * hfov
                cerca = min(porLabel.get(label, []), default=None,
                            key=lambda d: abs(d.bearing - rumbo))
                txt = nombreEs(label, None)
                c = color
                if cerca is not None and cerca.range_est is not None:
                    txt += f" {cerca.range_est:.1f}m"
                    if cerca.moving:
                        c = (60, 60, 255)
                cajas.append((x1, y1, x2, y2, txt, c))
        dibujarCajas(img, cajas)
        dibujarSensores(img, t, sens, pitch, heading, lecturas,
                        extra=f"pose {pose[0]:+.1f},{pose[1]:+.1f}")
        if self._aviso[1] and t - self._aviso[0] < 2.0:
            dibujarAviso(img, self._aviso[1])
        self.video.escribir(img)

    def cerrar(self):
        bridge, detectar, extras = self._restaurar
        bridge.detector.detect = detectar
        bridge._detectarExtras = extras
        if self.video is not None:
            self.video.cerrar()

    def reporte(self, usados):
        from nombres import nombreEs
        print("  -- lo que vio la vision (en cuantos cuadros; confianza max) --")
        for label, n in self.conteo.most_common():
            print(f"    {nombreEs(label, None):18s} {n:4d} de {usados}  "
                  f"({100 * n // max(1, usados):3d} %)  conf {self.confMax.get(label, 0):.2f}")
        voz = [a for a in self.alertas if a[1] >= 3]
        pips = [a for a in self.alertas if a[1] < 3]
        print(f"  -- alertas que habria dado: {len(voz)} con voz, "
              f"{len(pips)} pitidos de precaucion --")
        for t, nivel, txt in self.alertas[:40]:
            print(f"    t={t:6.1f}s  {'VOZ   ' if nivel >= 3 else 'pitido'}  {txt}")
        if self.video is not None:
            print(f"  video: {self.video.ruta}  ({self.video.cuadros} cuadros)")


def procesarSesion(carpeta, bridge, detPeligros, slam, mgr, args):
    import cv2
    import types
    import sensores as modSensores
    from sensores import SensoresGafas, ImuGafas
    from odometry import MotionEstimator
    from alertas import conectarUltrasonido
    from navigation_feed import buildNavigationFrame
    import montaje

    cuadros, lineas = leerSesion(carpeta)
    periodos = sorted(b[0] - a[0] for a, b in zip(cuadros, cuadros[1:]))
    fps = (1.0 / periodos[len(periodos) // 2] if periodos else 5.0) \
        / max(1, args.stride)
    repaso = _Repaso(carpeta, bridge, fps, getattr(args, "video", False))
    # pa todo lo que pregunta la hora (lecturas caducas, ¿va caminando?,
    # esperas de las alertas) el reloj es el de la GRABACION
    relojReal = modSensores.time
    modSensores.time = types.SimpleNamespace(
        time=lambda: repaso.reloj[0], sleep=relojReal.sleep)
    sens = SensoresGafas(iniciar=False)
    imu = ImuGafas(sens)
    repaso.mod.caminandoFn = sens.caminando
    est = None
    ecosCm = {"izq": [], "der": []}
    iLinea = 0
    tPrev = None
    usados = 0
    ultimosPeligros = []
    pitchs = []
    pitch = None
    t0 = relojReal.time()
    print(f"\n== {carpeta}: {len(cuadros)} cuadros, {len(lineas)} lineas "
          "del rp2040 ==")
    try:
        for k, (tCuadro, ruta) in enumerate(cuadros):
            # todo lo que mandaron los sensores hasta este cuadro
            while iLinea < len(lineas) and lineas[iLinea][0] <= tCuadro:
                t, texto = lineas[iLinea]
                repaso.reloj[0] = t
                sens.procesarLinea(texto, t)
                if texto.startswith("U,"):
                    partes = texto.split(",")
                    if len(partes) == 4 and partes[3].lstrip("-").isdigit():
                        ecosCm["izq" if partes[2] == "I" else "der"].append(
                            int(partes[3]))
                    # igual que en vivo: el eco pita desde su hilo
                    conectarUltrasonido(repaso.mod, sens.lecturasUS(t),
                                        pitchDeg=pitch)
                iLinea += 1
            repaso.reloj[0] = tCuadro
            if k % args.stride:
                continue
            frame = cv2.imread(ruta)
            if frame is None or not sens.imuOk:
                continue
            if est is None:
                # igual que en vivo: sin --step-length el paso es el
                # calibrado (o el promedio por cadencia); con el flag, fijo
                est = MotionEstimator(
                    imu=imu,
                    stepLengthM=args.step_length or montaje.LARGO_PASO_M,
                    porCadencia=args.step_length is None)
            dt = 0.2 if tPrev is None else max(1e-3, tCuadro - tPrev)
            tPrev = tCuadro
            usados += 1

            motion, odo = est.update(frame, dt)
            pitch = odo["pitch_deg"]
            if pitch is not None:
                pitchs.append(pitch)
                bridge.camPitchDeg = pitch
                if detPeligros is not None:
                    detPeligros.camPitchDeg = pitch
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and usados % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros
            usLecturas = validarConUltrasonido(
                detections, sens.lecturasUS(tCuadro), pitch)
            output = slam.step(motion,
                               fuseObservations(detections, usLecturas))
            pose = (output.x, output.y, output.theta)
            mgr.updateFromScan(pose, slam.observationsForMap()
                               + freeSpaceRays(usLecturas, maxLibreM=2.0)
                               + (detPeligros.tomarPisoLibre()
                                  if detPeligros else []),
                               maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            mgr.updatePosition(pose, slam.getLandmarks())
            navFrame = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom,
                ultrasonic=usLecturas)
            for prio, msg in repaso.mod.evaluar(navFrame)[0]:
                if repaso.mod.salida is None:
                    break
            repaso.contar(detections)
            e = imu.readEuler()
            repaso.dibujar(frame, tCuadro, sens, pitch,
                           math.degrees(e[0]) % 360 if e else None,
                           usLecturas, detections, pose, bridge.hfov)
            if usados % 50 == 0:
                print(f"  t={tCuadro:6.1f}s  pose=({output.x:+.1f}, "
                      f"{output.y:+.1f}, {math.degrees(output.theta):+.0f}°)  "
                      f"pasos={odo['steps_total']}  "
                      f"landmarks={len(slam.confirmedLandmarks())}  "
                      f"({usados / (relojReal.time() - t0):.1f} cuadros/s)",
                      flush=True)
    finally:
        modSensores.time = relojReal
        repaso.cerrar()
    repaso.reporte(usados)
    time_ = relojReal

    pasos = sens.pasos.stepsTotal
    print(f"  listo: {usados} cuadros en {time_.time() - t0:.0f} s")
    print("  -- reporte de sensores --")
    if args.step_length:
        print(f"  pasos contados: {pasos}  (x {args.step_length:.2f} m fijo = "
              f"{pasos * args.step_length:.1f} m caminados)")
    else:
        print(f"  pasos contados: {pasos}  -> {sens.metros:.1f} m caminados "
              "(largo de paso segun la cadencia)")
    try:
        from sensores import resumenCaminata, textoResumen
        r = resumenCaminata(os.path.join(carpeta, "serie.txt"))
        print("  caminata: " + textoResumen(r))
        e = r.get("eco")
        if e and e["metros"] >= 3.0 and abs(e["factor"] - 1.0) > 0.15:
            print("  ojo: el ultrasonido y los pasos difieren mas de 15 %: "
                  "conviene `python calibrar_gafas.py --solo paso`")
    except Exception as e:
        print(f"  (sin resumen de caminata: {e})")
    print(f"  muestras de imu: {sens.muestrasImu}  brincos de rumbo "
          f"ignorados: {sens.brincos}")
    if pitchs:
        pitchs.sort()
        print(f"  cabeceo: mediana {pitchs[len(pitchs) // 2]:+.1f}°, "
              f"de {pitchs[0]:+.1f}° a {pitchs[-1]:+.1f}°")
    for lado, cms in ecosCm.items():
        if not cms:
            print(f"  ultrasonico {lado}: sin lecturas")
            continue
        sinEco = sum(1 for c in cms if c >= 760)
        pegado = sum(1 for c in cms if 0 <= c <= 20)
        ausente = sum(1 for c in cms if c < 0)
        print(f"  ultrasonico {lado}: {len(cms)} lecturas, "
              f"{100 * sinEco / len(cms):.0f}% sin eco, "
              f"{100 * pegado / len(cms):.0f}% a 20 cm o menos, "
              f"{ausente} sin respuesta")
    return usados


def main():
    p = argparse.ArgumentParser(
        description="grabacion de las gafas -> mapa guardado")
    p.add_argument("sesiones", nargs="+",
                   help="carpeta(s) de grabar_sesion.py")
    p.add_argument("--map-dir", required=True,
                   help="carpeta del mapa (se acumula si ya existe)")
    p.add_argument("--fresh", action="store_true",
                   help="borrar el mapa previo y empezar de cero")
    p.add_argument("--stride", type=int, default=1,
                   help="procesar 1 de cada N cuadros grabados")
    p.add_argument("--step-length", type=float, default=None,
                   help="largo de paso (m); por defecto el calibrado")
    p.add_argument("--hazards", action="store_true",
                   help="detectar hoyos/coladeras/agua (SegFormer)")
    p.add_argument("--hazard-every", type=int, default=5)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--sin-extra", action="store_true")
    p.add_argument("--extra-every", type=int, default=4)
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--video", action="store_true",
                   help="guardar en la sesion video_sistema.mp4 con lo que "
                        "detecta el sistema, las distancias y las alertas")
    args = p.parse_args()

    if args.fresh and os.path.isdir(args.map_dir):
        import shutil
        shutil.rmtree(args.map_dir)
        print(f"[mapa] {args.map_dir} borrado (--fresh)")

    import montaje
    from vision_bridge import VisionBridge, GroundHazardDetector
    bridge = VisionBridge(modelPath=args.model, cameraProfile="picam3wide",
                          camHeightM=montaje.CAM_ALTURA_M,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    detPeligros = GroundHazardDetector(
        cameraProfile="picam3wide",
        camHeightM=montaje.CAM_ALTURA_M) if args.hazards else None

    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        print(f"[mapa] continuando mapa existente: {len(guardados)} landmarks")

    for carpeta in args.sesiones:
        procesarSesion(carpeta, bridge, detPeligros, slam, mgr, args)

    n = mgr.saveAll()
    mgr.saveLandmarks(slam.exportLandmarks())
    print(f"\n[mapa] {n} chunks + {len(slam.confirmedLandmarks())} landmarks "
          f"guardados en {args.map_dir}")
    croquis = mgr.renderCroquis(
        pose=None, landmarks=slam.confirmedLandmarks(),
        path=os.path.join(args.map_dir, "croquis.png"))
    print(f"[mapa] croquis: {croquis}  (revísalo: ¿se parece al lugar?)")
    from video_a_mapa import qualityReport
    qualityReport(args.map_dir)


if __name__ == "__main__":
    main()
