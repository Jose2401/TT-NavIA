"""
sesion_a_video.py — arma un video .mp4 de una grabacion de las gafas
(grabar_sesion.py guarda cuadros sueltos, no video) pa poder VERLA:

    python sesion_a_video.py sesiones/20261112_104724
        -> sesiones/20261112_104724/video.mp4   (rapido: imagen + sensores)

encima de la imagen va lo que mandaron los sensores en ese momento:
tiempo, pasos y metros, cabeceo, rumbo y la distancia de cada
ultrasonico. pa verlo ya con lo que DETECTA el sistema (cajas, distancias
y las alertas que habria dado) se usa el otro script, que corre la
vision completa y es mas lento:

    python sesion_a_mapa.py sesiones/20261112_104724 --map-dir maps/real/x --fresh --video
        -> sesiones/20261112_104724/video_sistema.mp4

se ven con vlc (ya esta en la pi):  vlc sesiones/20261112_104724/video.mp4
"""
import argparse
import math
import os
import shutil
import subprocess
import time


class VideoAnotado:
    """escribe cuadros bgr a un .mp4 (h264 con ffmpeg; si no hay ffmpeg,
    mp4v con opencv)."""

    def __init__(self, ruta, fps=5.0, ancho=640):
        self.ruta = ruta
        self.fps = fps
        self.ancho = ancho
        self._proc = None
        self._cv = None
        self._tam = None
        self.cuadros = 0

    def _abrir(self, w, h):
        self._tam = (w, h)
        if shutil.which("ffmpeg"):
            self._proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.fps),
                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "26", "-pix_fmt", "yuv420p", self.ruta],
                stdin=subprocess.PIPE)
        else:
            import cv2
            self._cv = cv2.VideoWriter(self.ruta,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, (w, h))

    def escribir(self, frameBgr):
        import cv2
        h0, w0 = frameBgr.shape[:2]
        w = self.ancho
        h = int(round(h0 * w / w0 / 2.0)) * 2
        if (w0, h0) != (w, h):
            frameBgr = cv2.resize(frameBgr, (w, h))
        if self._tam is None:
            self._abrir(w, h)
        if self._proc is not None:
            self._proc.stdin.write(frameBgr.tobytes())
        else:
            self._cv.write(frameBgr)
        self.cuadros += 1

    def cerrar(self):
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
        if self._cv is not None:
            self._cv.release()


def _texto(img, txt, x, y, color=(255, 255, 255), escala=0.42):
    import cv2
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala,
                color, 1, cv2.LINE_AA)


def _ascii(txt):
    """las fuentes de opencv no traen acentos."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", txt)
                   if unicodedata.category(c) != "Mn" and ord(c) < 128)


def dibujarSensores(img, t, sens, pitchDeg, headingDeg, lecturas,
                    extra=""):
    """la franja de arriba: lo que decian los sensores en ese cuadro."""
    ecos = {u.lado: (f"{u.range_m:.2f}" if u.valid else "---")
            for u in lecturas}
    cab = "  ?" if pitchDeg is None else f"{pitchDeg:+.0f}"
    rum = "  ?" if headingDeg is None else f"{headingDeg:.0f}"
    _texto(img, f"t {t:5.1f}s  pasos {sens.pasos.stepsTotal}  "
                f"{sens.metros:.1f} m  cabeceo {cab}  rumbo {rum}", 6, 14)
    _texto(img, f"ultrasonido izq {ecos.get('izq', ' ? ')} m   "
                f"der {ecos.get('der', ' ? ')} m   {extra}", 6, 30,
           color=(120, 230, 255))


def dibujarCajas(img, cajas):
    """cajas: [(x1, y1, x2, y2, etiqueta, color)] en coordenadas de img."""
    import cv2
    for x1, y1, x2, y2, etiqueta, color in cajas:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        _texto(img, _ascii(etiqueta), int(x1) + 3, max(44, int(y1) - 4),
               color=color)


def dibujarAviso(img, txt, color=(80, 80, 255)):
    h = img.shape[0]
    _texto(img, _ascii(txt)[:86], 6, h - 10, color=color, escala=0.46)


def videoRapido(carpeta, ancho=640):
    """imagen + sensores, sin correr la vision."""
    import cv2
    from sesion_a_mapa import leerSesion
    from sensores import SensoresGafas, ImuGafas
    cuadros, lineas = leerSesion(carpeta)
    if not cuadros:
        raise SystemExit(f"{carpeta}: no hay cuadros con imagen")
    periodos = sorted(b[0] - a[0] for a, b in zip(cuadros, cuadros[1:]))
    fps = 1.0 / periodos[len(periodos) // 2] if periodos else 5.0
    sens = SensoresGafas(iniciar=False)
    imu = ImuGafas(sens)
    video = VideoAnotado(os.path.join(carpeta, "video.mp4"), fps=fps,
                         ancho=ancho)
    i = 0
    for tCuadro, ruta in cuadros:
        base = time.time() - tCuadro
        while i < len(lineas) and lineas[i][0] <= tCuadro:
            sens.procesarLinea(lineas[i][1], base + lineas[i][0])
            i += 1
        frame = cv2.imread(ruta)
        if frame is None:
            continue
        frame = cv2.resize(frame, (ancho, int(frame.shape[0] * ancho
                                              / frame.shape[1] / 2) * 2))
        e = imu.readEuler()
        dibujarSensores(frame, tCuadro, sens,
                        math.degrees(e[2]) if e else None,
                        math.degrees(e[0]) % 360 if e else None,
                        sens.lecturasUS(base + tCuadro))
        video.escribir(frame)
    video.cerrar()
    return video.ruta, video.cuadros, fps


def main():
    p = argparse.ArgumentParser(
        description="grabacion de las gafas -> video.mp4 pa verla")
    p.add_argument("sesiones", nargs="+")
    p.add_argument("--ancho", type=int, default=640)
    args = p.parse_args()
    for carpeta in args.sesiones:
        ruta, n, fps = videoRapido(carpeta.rstrip("/"), args.ancho)
        print(f"{ruta}: {n} cuadros a {fps:.1f} por segundo "
              f"({n / fps:.0f} s)   ->   vlc {ruta}")


if __name__ == "__main__":
    main()
