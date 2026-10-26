import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from interfaces import MotionEstimate
from vision_bridge import fuseObservations

TIPS_GRABACION = """ de grabación (importan un buen, no hay odometría):
    - celular horizontal, a la altura del pecho/ojos, viendo al frente
    - camina y gira LENTO: el giro sale del flujo óptico y los
      barridos rápidos lo rompen
    - buena luz, nada de contraluces fuertes
    - pasa 2+ veces por el mismo lugar (confirma landmarks) y recorre
      cuartos completos, no trozos sueltos
    - con otro celular ajusta --hfov (gran angular de iphone ~100-120,
      lente normal ~65-75)
"""


class WalkDetector:
    def __init__(self, hfovDeg, workWidth=640, threshPx=1.4):
        self.hfov = math.radians(hfovDeg)
        self.workWidth = workWidth
        self.threshPx= threshPx
        self.prevGray = None

    def update(self, frameBgr, deltaYaw):
        import cv2
        import numpy as np
        h, w = frameBgr.shape[:2]
        escala = self.workWidth / float(w)
        gray = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.workWidth, int(h * escala)))
        prev, self.prevGray = self.prevGray, gray
        if prev is None:
            return False
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=120,
                                      qualityLevel=0.01, minDistance=14)
        if pts is None or len(pts) < 10:
            return False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        buenos = st.ravel() == 1
        if buenos.sum() < 10:
            return False
        fPx = (self.workWidth / 2.0) / math.tan(self.hfov / 2.0)
        dx = nxt[buenos, 0, 0] - pts[buenos, 0, 0] - fPx * math.tan(deltaYaw)
        dy =nxt[buenos, 0, 1] - pts[buenos, 0, 1]
        residuo = float(np.median(np.hypot(dx, dy)))
        return residuo > self.threshPx


# un video completo -> slam -> chunks
def processVideo(path, bridge, yawEst, detPeligros, slam, mgr, args):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = args.stride / max(1.0, fps)

    print(f"\n== {path}  ({total} frames @ {fps:.0f} fps, "
          f"procesando 1 de cada {args.stride}) ==")
    caminata = WalkDetector(math.degrees(bridge.hfov)) \
        if args.walk_speed > 0 else None
    frames = usados = caminados = 0
    ultimosPeligros = []
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if (frames - 1) % args.stride:
            continue
        usados += 1

        deltaRot, _ok = yawEst.update(frame)
        deltaTrans = 0.0
        if caminata is not None and caminata.update(frame, deltaRot):
            deltaTrans = args.walk_speed * dt
            caminados += 1
        detections = bridge.process(frame, egoYaw=deltaRot)
        if detPeligros is not None and usados % args.hazard_every == 0:
            ultimosPeligros = detPeligros.process(frame, detections)
        detections = detections + ultimosPeligros

        output = slam.step(MotionEstimate(deltaTrans, deltaRot, dt),
                           fuseObservations(detections, []))
        pose = (output.x, output.y, output.theta)
        mgr.updateFromScan(pose, slam.observationsForMap(),
                           maxRange=args.max_range)
        mgr.clearFootprint(output.x, output.y)
        mgr.updatePosition(pose, slam.getLandmarks())

        if usados % 100 == 0:
            el = time.time() - t0
            print(f"  f={frames}/{total}  theta="
                  f"{math.degrees(output.theta):+6.1f}  "
                  f"landmarks={len(slam.confirmedLandmarks())}  "
                  f"chunks={len(mgr.knownChunkIndices())}  "
                  f"({usados / el:.1f} fps de proceso)")
    cap.release()
    pct = 100 * caminados / max(1, usados)
    print(f"  listo: {usados} frames procesados en "
          f"{time.time() - t0:.0f} s (caminando el {pct:.0f}% del video)")


# checa si el mapa ya sirve pa entrenar
def qualityReport(mapDir):
    print("\n== Reporte de calidad del mapa ==")
    try:
        from navigation.real_maps import RealMapWorldFactory
        papu = RealMapWorldFactory(mapDir)
    except Exception as exc:
        print(f"  [X] El mapa AÚN NO sirve para entrenar: {exc}")
        print("      Graba más recorrido (más pasadas por los mismos "
              "lugares) y vuelve a correr este script con el nuevo "
              "video sobre el mismo --map-dir.")
        return False
    area = papu.free_area_m2()
    print(f"  [OK] {len(papu.statics)} celdas-obstáculo, "
          f"{len(papu.hazards)} celdas-peligro, "
          f"{area:.1f} m2 de espacio libre navegable")
    if area < 6.0:
        print("  [!] El área libre es pequeña: los episodios serán "
              "cortos. Graba más recorrido (pasa 2-3 veces por cada "
              "zona para que el mapa 'despeje' el piso).")
    import numpy as np
    rng = np.random.default_rng(0)
    _w, inicio, meta = papu.sample_episode(rng)
    print(f"  [OK] episodio de ejemplo: inicio=({inicio[0]:.1f},"
          f"{inicio[1]:.1f}) destino=({meta[0]:.1f},{meta[1]:.1f})")
    print("\n  Siguiente paso:")
    print(f"    python -m navigation.train --real-maps {mapDir} "
          "--timesteps 2000000 --out models/nav_ppo_real")
    return True


# arma todo y procesa los videos
def main():
    p = argparse.ArgumentParser(
        description="Video de recorrido (POV) -> mapa guardado para "
                    "entrenar la navegación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TIPS_GRABACION)
    p.add_argument("videos", nargs="+", help="archivo(s) de video")
    p.add_argument("--map-dir", required=True,
                   help="carpeta del mapa (se acumula si ya existe)")
    p.add_argument("--fresh", action="store_true",
                   help="borrar el mapa previo y empezar de cero")
    p.add_argument("--hfov", type=float, default=70.0,
                   help="FOV horizontal de la cámara en grados")
    p.add_argument("--vfov", type=float, default=None,
                   help="FOV vertical (por defecto ~0.78*hfov)")
    p.add_argument("--cam-height", type=float, default=1.55)
    p.add_argument("--cam-pitch", type=float, default=10.0)
    p.add_argument("--hazards", action="store_true",
                   help="detectar hoyos/coladeras/agua (SegFormer)")
    p.add_argument("--hazard-every", type=int, default=5)
    p.add_argument("--stride", type=int, default=2,
                   help="procesar 1 de cada N frames")
    p.add_argument("--walk-speed", type=float, default=1.1,
                   help="velocidad de caminata asumida (m/s) cuando el "
                        "flujo detecta que avanzas; 0 = desactivar "
                        "(ojo: la escala del mapa depende de esto)")
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--sin-extra", action="store_true",
                   help="apagar el detector extra (puertas/escaleras, oiv7)")
    p.add_argument("--extra-every", type=int, default=4)
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    args = p.parse_args()

    if args.fresh and os.path.isdir(args.map_dir):
        import shutil
        shutil.rmtree(args.map_dir)
        print(f"[mapa] {args.map_dir} borrado (--fresh)")

    from vision_bridge import (VisionBridge, VisualYawEstimator,
                               GroundHazardDetector)
    vfov = args.vfov if args.vfov else 0.78 * args.hfov
    print("Cargando módulo de visión (YOLOv8-seg)...")
    bridge = VisionBridge(modelPath=args.model, hfovDeg=args.hfov,
                          vfovDeg=vfov, camHeightM=args.cam_height,
                          camPitchDeg=args.cam_pitch,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    yawEst = VisualYawEstimator(hfovDeg=args.hfov)
    detPeligros = None
    if args.hazards:
        print("Cargando segmentación de escena (peligros de piso)...")
        detPeligros = GroundHazardDetector(
            hfovDeg=args.hfov, vfovDeg=vfov,
            camHeightM=args.cam_height, camPitchDeg=args.cam_pitch)

    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        print(f"[mapa] continuando mapa existente: {len(guardados)} landmarks, "
              f"{len(mgr.knownChunkIndices())} chunks")

    for video in args.videos:
        processVideo(video, bridge, yawEst, detPeligros, slam, mgr, args)

    n = mgr.saveAll()
    mgr.saveLandmarks(slam.exportLandmarks())
    print(f"\n[mapa] {n} chunks + "
          f"{len(slam.confirmedLandmarks())} landmarks guardados en "
          f"{args.map_dir}")
    croquis = mgr.renderCroquis(
        pose=None, landmarks=slam.confirmedLandmarks(),
        path=os.path.join(args.map_dir, "croquis.png"))
    print(f"[mapa] croquis: {croquis}  (revísalo: ¿se parece al lugar?)")

    qualityReport(args.map_dir)


if __name__ == "__main__":
    main()
