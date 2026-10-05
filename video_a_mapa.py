"""
video_a_mapa.py
Convierte un VIDEO DE RECORRIDO (grabado en primera persona con el
celular o con el hardware) en un MAPA GUARDADO listo para entrenar y
evaluar el modelo de navegación.

Qué hace, en orden:
    video -> visión (YOLOv8-seg + giro por flujo óptico
             [+ peligros de piso con --hazards])
          -> SLAM (EKF)
          -> mapa global por chunks (se guarda en --map-dir)
          -> reporte de calidad: ¿sirve para entrenar?

Después de esto:
    # entrenar/afinar el modelo mezclando tus mapas reales:
    python -m navigation.train --real-maps maps/casa --timesteps 2000000
    # o solo evaluar el modelo actual en tu entorno:
    python -m navigation.train --eval models/nav_ppo.zip --real-maps maps/casa

USO:
    python video_a_mapa.py recorrido.mp4 --map-dir maps/casa
    python video_a_mapa.py r1.mp4 r2.mp4 --map-dir maps/casa   # acumula
    python video_a_mapa.py recorrido.mp4 --map-dir maps/calle --hazards

CONSEJOS DE GRABACIÓN (importan mucho sin odometría):
    - Celular en horizontal, a la altura del pecho/ojos, apuntando al
      frente (como caminarías con los lentes).
    - Camina LENTO y gira LENTO (el giro se estima por flujo óptico;
      los barridos rápidos lo rompen).
    - Buena luz, evita contraluces fuertes.
    - Recorre el mismo lugar más de una vez (confirma landmarks).
    - Pasillos y cuartos completos > trozos sueltos.
    - Si usas otro celular/cámara ajusta --hfov (FOV horizontal
      aproximado del lente; el gran angular de un iPhone es ~100-120,
      el lente normal ~65-75).
"""
import argparse
import math
import os
import time

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from interfaces import MotionEstimate
from vision_bridge import fuse_observations


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
    frames = used = 0
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
        detections = bridge.process(frame)
        if hazard_det is not None and used % args.hazard_every == 0:
            last_hazards = hazard_det.process(frame, detections)
        detections = detections + last_hazards

        output = slam.step(MotionEstimate(0.0, delta_rot, dt),
                           fuse_observations(detections, []))
        pose = (output.x, output.y, output.theta)
        mgr.update_from_scan(pose, slam.observations_for_map(),
                             max_range=args.max_range)
        mgr.update_position(pose, slam.get_landmarks())

        if used % 100 == 0:
            el = time.time() - t0
            print(f"  f={frames}/{total}  theta="
                  f"{math.degrees(output.theta):+6.1f}  "
                  f"landmarks={len(slam.confirmed_landmarks())}  "
                  f"chunks={len(mgr.known_chunk_indices())}  "
                  f"({used / el:.1f} fps de proceso)")
    cap.release()
    print(f"  listo: {used} frames procesados en "
          f"{time.time() - t0:.0f} s")


def quality_report(map_dir):
    """¿El mapa resultante sirve como mundo de entrenamiento?"""
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
