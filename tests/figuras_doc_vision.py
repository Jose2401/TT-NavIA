"""
figuras del modulo de vision pa la doc del TT2: cajas de yolo (coco +
open images) con distancia y la segmentacion ade20k que usamos pa piso,
puertas y peligros. correr:
    python -m tests.figuras_doc_vision --out docs/TT_2026_B045/Capturas_TT2/vision
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from vision_bridge import VisionBridge, GroundHazardDetector

ESCENAS = ((11, 0.45, "entrada de la sala de maestros"),
           (14, 0.45, "oficina con una persona caminando"),
           (6, 0.45, "calle con baches"),
           (16, 0.45, "piso con un tapete"))

COLOR_SEG = {"piso": (80, 200, 80), "puerta": (230, 120, 30),
             "agua": (230, 200, 0), "escalera": (60, 60, 230),
             "pared": (170, 170, 170)}


def _frameDe(video, frac):
    cap = cv2.VideoCapture(video)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # unos cuadros antes pa que el rastreador tenga historia
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac) - 12))
    frames = []
    for _ in range(13):
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    return frames


def _overlaySeg(frame, hz):
    small = cv2.resize(frame, (480, int(round(frame.shape[0] * 480 /
                                              frame.shape[1]))))
    seg = hz._segmenter.segment(small)
    walk, water, stairs = hz._segmenter.extract_ground_masks(seg)
    door = np.isin(seg, hz._segmenter.DOOR_IDS)
    wall = np.isin(seg, hz._segmenter.WALL_IDS)
    color = small.copy()
    for mask, nom in ((wall, "pared"), (walk, "piso"), (door, "puerta"),
                      (water, "agua"), (stairs, "escalera")):
        color[mask.astype(bool)] = COLOR_SEG[nom]
    return cv2.addWeighted(small, 0.45, color, 0.55, 0)


def _dibujarDets(frame, dets, hfov):
    img = cv2.resize(frame, (960, int(round(frame.shape[0] * 960 /
                                            frame.shape[1]))))
    h, w = img.shape[:2]
    ocupados = []
    for d in sorted(dets, key=lambda d: -d.bearing):
        col = int(w / 2.0 - (d.bearing / hfov) * w)
        fila = 0
        while any(abs(col - c) < 190 and f == fila for c, f in ocupados):
            fila += 1
        ocupados.append((col, fila))
        yTxt = int(h * 0.12) + 30 * fila
        if d.hazard:
            c = (255, 0, 255)
        elif d.moving:
            c = (0, 0, 255)
        else:
            c = (0, 220, 0)
        cv2.line(img, (col, yTxt + 8), (col, int(h * 0.8)), c, 3)
        tag = d.label
        if d.estado:
            tag += f" ({d.estado})"
        if d.range_est is not None:
            tag += f" {d.range_est:.1f} m"
        if d.moving:
            tag += " [movil]"
        for g, cc in ((5, (0, 0, 0)), (2, c)):
            cv2.putText(img, tag, (min(max(0, col - 70), w - 230), yTxt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cc, g, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/TT_2026_B045/Capturas_TT2/vision")
    p.add_argument("--hfov", type=float, default=100.0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    videos = sorted(glob.glob("videosxd/*"))

    bridge = VisionBridge(hfovDeg=args.hfov, extraEvery=1)
    hz = GroundHazardDetector(hfovDeg=args.hfov)
    hz._cargarModelo()
    filas = []
    paneles = []
    for idx, frac, nombre in ESCENAS:
        frames = _frameDe(videos[idx], frac)
        bridge.tracker.__init__()
        for fr in frames[:-1]:
            bridge.process(fr)
        fr = frames[-1]
        t0 = time.perf_counter()
        dets = bridge.process(fr)
        t1 = time.perf_counter()
        pel = hz.process(fr, dets)
        t2 = time.perf_counter()
        todas = dets + pel
        izq = _dibujarDets(fr, todas, bridge.hfov)
        der = _overlaySeg(fr, hz)
        der = cv2.resize(der, (int(der.shape[1] * izq.shape[0] /
                                   der.shape[0]), izq.shape[0]))
        panel = np.hstack([izq, der])
        cv2.imwrite(os.path.join(args.out, f"vision_escena{idx}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        paneles.append(panel)
        filas.append(dict(escena=nombre, video=os.path.basename(videos[idx]),
                          ms_yolo_coco_oiv7=1000 * (t1 - t0),
                          ms_segformer=1000 * (t2 - t1),
                          detecciones=[dict(label=d.label, estado=d.estado,
                                            m=round(d.range_est, 2),
                                            movil=d.moving, peligro=d.hazard)
                                       for d in todas]))
        print(filas[-1])

    # tiempos por modelo, promedio de 10 corridas sobre el mismo cuadro
    fr = cv2.resize(frames[-1], (640, 360))
    tm = {}
    for nom, fn in (("yolov8s_seg_coco", lambda: bridge.detector.detect(fr)),
                    ("yolov8s_oiv7", lambda: bridge._detectarExtras(fr)),
                    ("segformer_b0", lambda: hz.process(frames[-1], []))):
        fn()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        tm[nom] = 1000 * (time.perf_counter() - t0) / 10
    print(tm)
    json.dump({"escenas": filas, "ms_por_modelo": tm},
              open(os.path.join(args.out, "vision_eval.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
