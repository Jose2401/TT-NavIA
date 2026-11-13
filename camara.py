"""
camara.py — captura de video pa los mains.

en la pi la Camera Module 3 Wide NO abre con cv2.VideoCapture (va por
libcamera): se usa picamera2. ojo con dos cosas del prototipo
(docs/GAFAS_HARDWARE.md):
  - la camara va montada AL REVES dentro del frente: hay que girar la
    imagen 180° (Transform hflip+vflip), el sistema no lo corrige
  - el modo de 1536x864 del sensor es un RECORTE (pierde campo de
    vision); se pide el de 2304x1296, que usa el sensor completo, y el
    isp lo baja al tamaño de trabajo. asi si valen los 102° del perfil
    "picam3wide"

un hilo se queda con el cuadro mas reciente; read() regresa (ok, bgr)
igual que cv2, pa que los mains no cambien. sin picamera2 (laptop) cae
a cv2.VideoCapture.
"""
import os
import threading
import time


class CamaraPi:
    MODO_SENSOR = (2304, 1296)      # sensor completo con binning 2x2
    INTENTOS = 3
    ESPERA_PRIMER_CUADRO_S = 6.0

    def __init__(self, ancho=1280, alto=720, girar180=True,
                 fpsMax=30.0, lente=1.0):
        """lente: posicion fija del enfoque en dioptrias (1.0 = 1 m; a
        la resolucion de trabajo queda nitido de ~30 cm a infinito y no
        anda 'cazando' foco al caminar)."""
        self.ok = False
        self.razon = ""
        self.detectada = False        # el sensor contesta (aunque no de imagen)
        self._cam = None
        self._frame = None
        self._n = 0
        self._nLeido = 0
        self.tFrame = 0.0
        self._cond = threading.Condition()
        self._parar = False
        try:
            os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")
            from picamera2 import Picamera2
            from libcamera import Transform, controls
        except ImportError as e:
            self.razon = f"picamera2 no disponible ({e})"
            return
        import logging
        Picamera2.set_logging(logging.ERROR)
        try:
            self.detectada = bool(Picamera2.global_camera_info())
        except Exception:
            self.detectada = False
        if not self.detectada:
            self.razon = "no se detectó ninguna cámara csi"
            return
        # recien prendida la pi la camara a veces esta ocupada un momento
        # (pipewire la revisa al arrancar): se reintenta antes de rendirse
        for intento in range(self.INTENTOS):
            self._cam = None
            try:
                self._cam = Picamera2()
                giro = Transform(hflip=1, vflip=1) if girar180 else Transform()
                cfg = self._cam.create_video_configuration(
                    main={"size": (ancho, alto), "format": "RGB888"},
                    raw={"size": self.MODO_SENSOR},
                    transform=giro, buffer_count=4)
                self._cam.configure(cfg)
                ctrl = {"FrameDurationLimits": (int(1e6 / fpsMax), 66666)}
                disponibles = self._cam.camera_controls
                if "AfMode" in disponibles:
                    ctrl["AfMode"] = controls.AfModeEnum.Manual
                    ctrl["LensPosition"] = float(lente)
                if "AeExposureMode" in disponibles:
                    # exposicion corta: menos barrido al caminar
                    ctrl["AeExposureMode"] = controls.AeExposureModeEnum.Short
                self._cam.set_controls(ctrl)
                self._cam.start()
                self.razon = ""
                break
            except Exception as e:
                self.razon = f"no se pudo iniciar la cámara ({e})"
                self._cerrarCam()
                if intento + 1 < self.INTENTOS:
                    time.sleep(2.0)
        else:
            return
        threading.Thread(target=self._loop, daemon=True).start()
        # que se detecte no basta: si el cable plano esta flojo el sensor
        # contesta pero la imagen nunca llega (libcamera: "frontend has
        # timed out"). sin el primer cuadro, no hay camara
        with self._cond:
            if self._frame is None:
                self._cond.wait(self.ESPERA_PRIMER_CUADRO_S)
            hayCuadro = self._frame is not None
        if not hayCuadro:
            self.razon = ("la cámara se detecta pero no entrega imagen: "
                          "revisar que el cable plano esté bien metido en "
                          "la cámara y en la pi")
            self._parar = True
            threading.Thread(target=self._cerrarCam, daemon=True).start()
            return
        self.ok = True

    def _cerrarCam(self):
        try:
            if self._cam is not None:
                self._cam.stop()
                self._cam.close()
        except Exception:
            pass

    def _loop(self):
        while not self._parar:
            try:
                # "RGB888" de picamera2 ya viene en orden BGR (opencv)
                frame = self._cam.capture_array("main")
            except Exception as e:
                self.razon = f"captura falló ({e})"
                time.sleep(0.2)
                continue
            with self._cond:
                self._frame = frame
                self._n += 1
                self.tFrame = time.time()
                self._cond.notify_all()

    def isOpened(self):
        return self.ok

    def sinImagen(self, segundos=5.0):
        """True si lleva ese rato sin llegar un cuadro (cable flojo a
        media caminata): el main lo avisa por voz."""
        return self.ok and time.time() - self.tFrame > segundos

    def read(self, esperaS=1.0):
        """el cuadro mas reciente; espera a que haya uno NUEVO pa no
        procesar dos veces el mismo."""
        with self._cond:
            if self._n == self._nLeido:
                self._cond.wait(esperaS)
            if self._frame is None or self._n == self._nLeido:
                return False, None
            self._nLeido = self._n
            return True, self._frame

    def release(self):
        self._parar = True
        self._cerrarCam()


def prepararGui():
    """el opencv de pip trae su Qt sin fuentes y llena la terminal de
    avisos: se le apunta a las del sistema si existen."""
    try:
        import cv2  # noqa: F401  (al importarse pone su propia ruta)
    except ImportError:
        return
    if os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
        return
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype"):
        if os.path.isdir(d):
            os.environ["QT_QPA_FONTDIR"] = d
            return


def gafasConectadas():
    """True si esta el rp2040 de las gafas por usb (pa que los mains de
    laptop usen solos la imu y los ultrasonicos cuando corren en la pi)."""
    try:
        from sensores import buscarPuerto
        return buscarPuerto() is not None
    except Exception:
        return False


def abrirCamara(fuente=0, ancho=1280, alto=720):
    """-> (captura, perfil). fuente: indice de camara o ruta de video.
    con un indice en la pi usa la camara csi de las gafas (perfil
    'picam3wide'); si no hay, o es un video, cv2.VideoCapture
    (perfil 'laptop')."""
    esVideo = isinstance(fuente, str) and not fuente.isdigit()
    if not esVideo:
        cam = CamaraPi(ancho=ancho, alto=alto)
        if cam.ok or cam.detectada:
            # ojo: si la camara csi esta pero fallo, NO se cae a
            # cv2.VideoCapture: en la pi /dev/video0 "abre" y nunca da
            # un cuadro, y el main corria a ciegas sin avisar
            return cam, "picam3wide"
    import cv2
    cap = cv2.VideoCapture(fuente if esVideo else int(fuente))
    return cap, "laptop"


if __name__ == "__main__":
    # prueba rapida: python camara.py [foto.jpg]
    import sys
    cam, perfil = abrirCamara()
    if not cam.isOpened():
        raise SystemExit(f"sin cámara: {getattr(cam, 'razon', '')}")
    time.sleep(1.0)
    t0, n = time.time(), 0
    while time.time() - t0 < 2.0:
        ok, frame = cam.read()
        n += ok
    print(f"perfil {perfil}: {n / 2.0:.1f} fps, cuadro {frame.shape}")
    if len(sys.argv) > 1:
        import cv2
        cv2.imwrite(sys.argv[1], frame)
        print("guardado", sys.argv[1])
    cam.release()
