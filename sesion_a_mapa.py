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


def leerSesion(carpeta):
    """-> (cuadros [(t, ruta)], lineas [(t, texto)]) ordenados por t."""
    cuadros = []
    with open(os.path.join(carpeta, "cuadros.csv"), encoding="utf-8") as fh:
        next(fh)
        for renglon in fh:
            i, t = renglon.strip().split(",")
            cuadros.append((float(t), os.path.join(carpeta, "cuadros",
                                                   f"{int(i):06d}.jpg")))
    lineas = []
    with open(os.path.join(carpeta, "serie.txt"), encoding="utf-8") as fh:
        for renglon in fh:
            t, _, texto = renglon.strip().partition(" ")
            try:
                lineas.append((float(t), texto))
            except ValueError:
                pass
    return cuadros, lineas


def procesarSesion(carpeta, bridge, detPeligros, slam, mgr, args):
    import cv2
    from sensores import SensoresGafas, ImuGafas
    from odometry import MotionEstimator
    import montaje

    cuadros, lineas = leerSesion(carpeta)
    sens = SensoresGafas(iniciar=False)
    imu = ImuGafas(sens)
    est = None
    ecosCm = {"izq": [], "der": []}
    iLinea = 0
    tPrev = None
    usados = 0
    ultimosPeligros = []
    pitchs = []
    t0 = time.time()
    print(f"\n== {carpeta}: {len(cuadros)} cuadros, {len(lineas)} lineas "
          "del rp2040 ==")
    for k, (tCuadro, ruta) in enumerate(cuadros):
        # todo lo que mandaron los sensores hasta este cuadro. el reloj
        # de "ahora" es el de la grabacion pa que nada salga caducado
        base = time.time() - tCuadro
        while iLinea < len(lineas) and lineas[iLinea][0] <= tCuadro:
            t, texto = lineas[iLinea]
            sens.procesarLinea(texto, base + t)
            if texto.startswith("U,"):
                partes = texto.split(",")
                if len(partes) == 4 and partes[3].lstrip("-").isdigit():
                    ecosCm["izq" if partes[2] == "I" else "der"].append(
                        int(partes[3]))
            iLinea += 1
        if k % args.stride:
            continue
        frame = cv2.imread(ruta)
        if frame is None or not sens.imuOk:
            continue
        if est is None:
            # igual que en vivo: sin --step-length el paso va segun la
            # cadencia; con el flag, fijo
            est = MotionEstimator(
                imu=imu, stepLengthM=args.step_length or montaje.LARGO_PASO_M,
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
            detections, sens.lecturasUS(base + tCuadro), pitch)
        output = slam.step(motion, fuseObservations(detections, usLecturas))
        pose = (output.x, output.y, output.theta)
        mgr.updateFromScan(pose, slam.observationsForMap()
                           + freeSpaceRays(usLecturas, maxLibreM=2.0)
                           + (detPeligros.tomarPisoLibre() if detPeligros else []),
                           maxRange=args.max_range)
        mgr.clearFootprint(output.x, output.y)
        mgr.updatePosition(pose, slam.getLandmarks())
        if usados % 50 == 0:
            print(f"  t={tCuadro:6.1f}s  pose=({output.x:+.1f}, {output.y:+.1f}, "
                  f"{math.degrees(output.theta):+.0f}°)  "
                  f"pasos={odo['steps_total']}  "
                  f"landmarks={len(slam.confirmedLandmarks())}  "
                  f"({usados / (time.time() - t0):.1f} cuadros/s)")

    pasos = sens.pasos.stepsTotal
    print(f"  listo: {usados} cuadros en {time.time() - t0:.0f} s")
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
