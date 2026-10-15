"""
video_a_mapa.py — convierte un video de recorrido (pov con el celular)
en un mapa guardado listo para entrenar/evaluar la navegación.
video -> visión + flujo óptico -> ekf slam -> chunks -> reporte de calidad.

    python video_a_mapa.py recorrido.mp4 --map-dir maps/casa
    python video_a_mapa.py r1.mp4 r2.mp4 --map-dir maps/casa   # acumula
    python -m navigation.train --real-maps maps/casa           # después

CONSEJOS de grabación (importan un buen, no hay odometría):
    - celular horizontal, a la altura del pecho/ojos, viendo al frente
    - camina y gira LENTO: el giro sale del flujo óptico y los
      barridos rápidos lo rompen
    - buena luz, nada de contraluces fuertes
    - pasa 2+ veces por el mismo lugar (confirma landmarks) y recorre
      cuartos completos, no trozos sueltos
    - con otro celular ajusta --hfov (gran angular de iphone ~100-120,
      lente normal ~65-75)
"""
import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from interfaces import MotionEstimate
from vision_bridge import fuse_observations


class WalkDetector:
    """detecta si la persona esta CAMINANDO viendo el flujo optico.
    ojo: un video no trae odometria, asi que sin esto la pose solo gira
    y el mapa queda sin piso despejado (inservible pa entrenar). la idea:
    quitamos al flujo el corrimiento que causa el giro y si lo que sobra
    se mueve (la escena 'fluye' hacia ti), ps estas caminando y avanzamos
    a --walk-speed. la escala depende de esa velocidad, no es exacta."""

    def __init__(self, hfov_deg, work_width=640, thresh_px=1.4):
        self.hfov = math.radians(hfov_deg)
        self.work_width = work_width
        self.thresh_px = thresh_px
        self.prev_gray = None

    def update(self, frame_bgr, delta_yaw):
        import cv2
        import numpy as np
        h, w = frame_bgr.shape[:2]
        scale = self.work_width / float(w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.work_width, int(h * scale)))
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return False
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=120,
                                      qualityLevel=0.01, minDistance=14)
        if pts is None or len(pts) < 10:
            return False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = st.ravel() == 1
        if good.sum() < 10:
            return False
        # el giro mueve toda la escena en x; se lo restamos y lo que
        # quede de flujo es traslacion de verdad
        f_px = (self.work_width / 2.0) / math.tan(self.hfov / 2.0)
        dx = nxt[good, 0, 0] - pts[good, 0, 0] - f_px * math.tan(delta_yaw)
        dy = nxt[good, 0, 1] - pts[good, 0, 1]
        residual = float(np.median(np.hypot(dx, dy)))
        return residual > self.thresh_px


def process_video(path, bridge, yaw_est, hazard_det, slam, mgr, args):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = args.stride / max(1.0, fps)

    print(f"\n== {path}  ({total} frames @ {fps:.0f} fps, "
          f"procesando 1 de cada {args.stride}) ==")
    walk = WalkDetector(math.degrees(bridge.hfov)) \
        if args.walk_speed > 0 else None
    frames = used = walked = 0
    last_hazards = []
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if (frames - 1) % args.stride:
            continue
        used += 1

        delta_rot, _ok = yaw_est.update(frame)
        # avance: si el flujo dice que caminas, avanzamos a walk_speed
        delta_trans = 0.0
        if walk is not None and walk.update(frame, delta_rot):
            delta_trans = args.walk_speed * dt
            walked += 1
        detections = bridge.process(frame, ego_yaw=delta_rot)
        if hazard_det is not None and used % args.hazard_every == 0:
            last_hazards = hazard_det.process(frame, detections)
        detections = detections + last_hazards

        output = slam.step(MotionEstimate(delta_trans, delta_rot, dt),
                           fuse_observations(detections, []))
        pose = (output.x, output.y, output.theta)
        mgr.update_from_scan(pose, slam.observations_for_map(),
                             max_range=args.max_range)
        mgr.clear_footprint(output.x, output.y)   # por donde pasas es libre
        mgr.update_position(pose, slam.get_landmarks())

        if used % 100 == 0:
            el = time.time() - t0
            print(f"  f={frames}/{total}  theta="
                  f"{math.degrees(output.theta):+6.1f}  "
                  f"landmarks={len(slam.confirmed_landmarks())}  "
                  f"chunks={len(mgr.known_chunk_indices())}  "
                  f"({used / el:.1f} fps de proceso)")
    cap.release()
    pct = 100 * walked / max(1, used)
    print(f"  listo: {used} frames procesados en "
          f"{time.time() - t0:.0f} s (caminando el {pct:.0f}% del video)")


def quality_report(map_dir):
    """¿el mapa ya sirve como mundo de entrenamiento?"""
    print("\n== Reporte de calidad del mapa ==")
    try:
        from navigation.real_maps import RealMapWorldFactory
        f = RealMapWorldFactory(map_dir)
    except Exception as exc:
        print(f"  [X] El mapa AÚN NO sirve para entrenar: {exc}")
        print("      Graba más recorrido (más pasadas por los mismos "
              "lugares) y vuelve a correr este script con el nuevo "
              "video sobre el mismo --map-dir.")
        return False
    area = f.free_area_m2()
    print(f"  [OK] {len(f.statics)} celdas-obstáculo, "
          f"{len(f.hazards)} celdas-peligro, "
          f"{area:.1f} m2 de espacio libre navegable")
    if area < 6.0:
        print("  [!] El área libre es pequeña: los episodios serán "
              "cortos. Graba más recorrido (pasa 2-3 veces por cada "
              "zona para que el mapa 'despeje' el piso).")
    import numpy as np
    rng = np.random.default_rng(0)
    _w, start, goal = f.sample_episode(rng)
    print(f"  [OK] episodio de ejemplo: inicio=({start[0]:.1f},"
          f"{start[1]:.1f}) destino=({goal[0]:.1f},{goal[1]:.1f})")
    print("\n  Siguiente paso:")
    print(f"    python -m navigation.train --real-maps {map_dir} "
          "--timesteps 2000000 --out models/nav_ppo_real")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Video de recorrido (POV) -> mapa guardado para "
                    "entrenar la navegación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("CONSEJOS")[1])
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
    bridge = VisionBridge(model_path=args.model, hfov_deg=args.hfov,
                          vfov_deg=vfov, cam_height_m=args.cam_height,
                          cam_pitch_deg=args.cam_pitch)
    yaw_est = VisualYawEstimator(hfov_deg=args.hfov)
    hazard_det = None
    if args.hazards:
        print("Cargando segmentación de escena (peligros de piso)...")
        hazard_det = GroundHazardDetector(
            hfov_deg=args.hfov, vfov_deg=vfov,
            cam_height_m=args.cam_height, cam_pitch_deg=args.cam_pitch)

    mgr = ChunkedMapManager(map_dir=args.map_dir,
                            chunk_size_m=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    saved = mgr.load_landmarks()
    if saved:
        slam.import_landmarks(saved)
        print(f"[mapa] continuando mapa existente: {len(saved)} landmarks, "
              f"{len(mgr.known_chunk_indices())} chunks")

    for video in args.videos:
        process_video(video, bridge, yaw_est, hazard_det, slam, mgr, args)

    n = mgr.save_all()
    mgr.save_landmarks(slam.export_landmarks())
    print(f"\n[mapa] {n} chunks + "
          f"{len(slam.confirmed_landmarks())} landmarks guardados en "
          f"{args.map_dir}")
    croquis = mgr.render_croquis(
        pose=None, landmarks=slam.confirmed_landmarks(),
        path=os.path.join(args.map_dir, "croquis.png"))
    print(f"[mapa] croquis: {croquis}  (revísalo: ¿se parece al lugar?)")

    quality_report(args.map_dir)


if __name__ == "__main__":
    main()
