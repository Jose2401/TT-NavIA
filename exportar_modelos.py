"""
exportar_modelos.py — deja los dos yolo en formato ncnn pa la pi.

medido en la raspberry pi 5 (2026-11-10, cuadro de 640x360):
    yolov8s-seg   pytorch 1390 ms  ->  ncnn 370 ms
    yolov8s-oiv7  pytorch  740 ms  ->  ncnn 225 ms
son los MISMOS pesos (no se reentrena nada); las detecciones salen
iguales salvo diferencias de 0.01-0.02 en la confianza. vision_bridge
usa solo la carpeta *_ncnn_model si existe (NAVIA_SIN_NCNN=1 pa forzar
el .pt).

uso:  python exportar_modelos.py            (una vez, tarda ~1 min)
"""
import os
import sys

VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vision")
# alto x ancho: un cuadro 16:9 de 640 de ancho, redondeado a multiplo de 32
IMGSZ = (384, 640)


def exportar(base, forzar=False):
    from ultralytics import YOLO
    pt = os.path.join(VISION_DIR, base + ".pt")
    destino = os.path.join(VISION_DIR, base + "_ncnn_model")
    if not os.path.exists(pt):
        print(f"no esta {pt}, se salta")
        return None
    if os.path.isdir(destino) and not forzar:
        print(f"ya existe {destino} (--forzar pa rehacerlo)")
        return destino
    ruta = YOLO(pt).export(format="ncnn", imgsz=IMGSZ, verbose=False)
    print(f"listo: {ruta}")
    return ruta


if __name__ == "__main__":
    forzar = "--forzar" in sys.argv
    for base in ("yolov8s-seg", "yolov8s-oiv7"):
        exportar(base, forzar)
