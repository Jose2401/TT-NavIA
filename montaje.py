"""
montaje.py — geometria fisica de los lentes segun el CAD (docs/Render,
VistaPrincipal, VistaReal). una sola fuente de verdad pa todos los
modulos que necesitan saber donde y hacia donde ve cada sensor.

  - camara: centro del frente, sobre el puente de la nariz (origen).
    va montada AL REVES (camara.py gira la imagen 180°)
  - 2 ultrasonicos MB1232: en el FRENTE, uno sobre cada lente,
    apuntando HACIA ADELANTE (paralelos a la camara), separados
    US_SEPARACION_M. ojo: no son laterales.
  - imu bno055: DE PIE en el puente de arriba (el firmware del rp2040
    le remapea los ejes). todo va en el mismo frente rigido, asi que
    el pitch de la imu = pitch de la camara = pitch de los ultrasonicos.
  - la imu y los ultrasonicos NO van al i2c de la pi: los lee el
    RP2040-Zero de las gafas y llegan por usb (sensores.py). el boton
    y su led si van a la pi (GPIO17 y GPIO27). docs/GAFAS_HARDWARE.md

convencion del sistema: x adelante, y a la IZQUIERDA, bearing + = izq.
medir con regla en el prototipo real y ajustar estos valores. lo que
cambia con la PERSONA (altura de los ojos y largo del paso) se guarda
en calibracion_usuario.json con `python calibrar_gafas.py`.
"""
import json
import math
import os

US_SEPARACION_M = 0.10            # entre centros de los dos ultrasonicos
# toe-out de cada uno (0 = paralelos). el ez3 es de haz ANGOSTO: si
# quieren algo de lateral se pueden abrir ~15° c/u (cubre ±30° total)
# a cambio de perder el traslape del centro (y el estereo)
US_APERTURA_DEG = 0.0
# del datasheet mb1232 (pd11848, grafica del patron en cuadros de
# 30 cm): ~±14° pa objetos medianos; no sirve pa laterales
US_MEDIO_CONO_DEG = 15.0
US_MAX_M = 7.65                   # lo que reporta cuando NO hay eco
US_MIN_M = 0.20                   # reporta 20 cm pa todo lo que este MAS cerca
US_VOLTS = 3.3                    # van a 3.3 V (el rp2040 no aguanta 5): menos alcance
CAM_ALTURA_M = 1.55               # altura de los ojos/camara al piso
# largo de paso. SIN calibrar: el de un adulto promedio, 0.68 m a paso
# normal (~105 pasos/min), acortado en proporcion a la cadencia (largoPaso
# de abajo). CALIBRADO (calibracion_usuario.json): el de esa persona,
# fijo. con las dos caminatas reales del 2026-11-10 el paso del usuario
# salio de 0.43-0.48 m sin importar la cadencia (68 a 100 pasos/min)
LARGO_PASO_M = 0.68
CADENCIA_PASO_PPM = 105.0         # cadencia a la que el paso promedio mide eso
PASO_CALIBRADO = False            # True = LARGO_PASO_M es de quien trae los lentes

CALIBRACION_USUARIO = "calibracion_usuario.json"


def cargarUsuario(ruta=CALIBRACION_USUARIO):
    """altura de la camara y largo de paso de quien trae los lentes
    (los mide calibrar_gafas.py). sin archivo quedan los de arriba."""
    global CAM_ALTURA_M, LARGO_PASO_M, PASO_CALIBRADO
    try:
        with open(ruta, encoding="utf-8") as fh:
            c = json.load(fh)
    except (OSError, ValueError):
        return {}
    if 1.0 <= float(c.get("alturaCamaraM", 0)) <= 2.2:
        CAM_ALTURA_M = float(c["alturaCamaraM"])
    if 0.25 <= float(c.get("largoPasoM", 0)) <= 1.1:
        LARGO_PASO_M = float(c["largoPasoM"])
        PASO_CALIBRADO = True
    return c


def largoPaso(cadenciaPpm=None):
    """cuanto avanza un paso. calibrado: el largo de esa persona, fijo
    (medido por ultrasonido caminando hacia una pared; se afina solo con
    cada corrida, sensores.SensoresGafas.aprenderPaso). sin calibrar: el
    promedio de un adulto, proporcional a la cadencia (la gente acorta
    el paso al ir despacio; con 0.68 fijo una caminata lenta de 4.6 m
    salia de 6.8 m y asi de 4.8), recortado pa que un hueco raro entre
    pasos no lo dispare."""
    if PASO_CALIBRADO or not cadenciaPpm:
        return LARGO_PASO_M
    f = cadenciaPpm / CADENCIA_PASO_PPM
    return LARGO_PASO_M * max(0.45, min(1.30, f))


def guardarUsuario(ruta=CALIBRACION_USUARIO, **valores):
    c = {}
    try:
        with open(ruta, encoding="utf-8") as fh:
            c = json.load(fh)
    except (OSError, ValueError):
        pass
    c.update(valores)
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())      # la pi se apaga desconectandola
    cargarUsuario(ruta)
    return c


cargarUsuario()

# alcance maximo por tamaño del objetivo (datasheet, graficas A-D). el
# ez3 "filtra lo chico": una botella o una pata de silla casi no la ve
_ALCANCE_5V = {"chico": 1.5, "persona": 2.4, "mediano": 3.6, "grande": 7.5}
_ALCANCE_33V = {"chico": 1.2, "persona": 1.8, "mediano": 2.7, "grande": 6.0}
_TAMANO = {
    "person": "persona", "dog": "persona", "cat": "chico",
    "bottle": "chico", "cup": "chico", "cell phone": "chico",
    "vase": "chico", "book": "chico", "backpack": "chico",
    "handbag": "chico", "laptop": "chico", "mouse": "chico",
    "chair": "mediano", "potted plant": "mediano", "monitor": "mediano",
    "trash can": "mediano", "bench": "mediano", "fire hydrant": "mediano",
    "table": "mediano", "desk": "mediano", "toilet": "mediano",
    "door": "grande", "fridge": "grande", "cabinet": "grande",
    "shelf": "grande", "sofa": "grande", "bed": "grande",
    "whiteboard": "grande", "window": "grande", "car": "grande",
    "truck": "grande", "bus": "grande", "counter": "grande",
}


def alcanceUS(label, volts=None):
    """hasta donde el mb1232 alcanza a ver un objeto de esa clase."""
    tabla = _ALCANCE_33V if (volts or US_VOLTS) < 4.0 else _ALCANCE_5V
    return tabla[_TAMANO.get(label, "mediano")]


def usPosicion(lado):
    """-> (offsetY, bearing) del sensor 'izq' o 'der' en el marco del
    lente (y + = izquierda, bearing + = izquierda)."""
    s = 1.0 if lado == "izq" else -1.0
    return (s * US_SEPARACION_M / 2.0,
            s * math.radians(US_APERTURA_DEG))


def usAlCentro(rango, bearing, offsetY):
    """pasa una lectura de un sensor desplazado a polares desde el
    centro (la camara), que es donde vive todo lo demas."""
    px = rango * math.cos(bearing)
    py = offsetY + rango * math.sin(bearing)
    return math.hypot(px, py), math.atan2(py, px)


def bearingEstereo(rIzq, rDer, sep=US_SEPARACION_M):
    """con los dos frontales viendo el MISMO objeto, la diferencia de
    rangos dice de que lado esta: y = (rDer² - rIzq²) / (2·sep).
    sirve mas de cerca (a 1 cm de resolucion y 10 cm de base, a 2 m
    da ~6 grados); se recorta al cono del haz."""
    r = min(rIzq, rDer)
    if r <= 0.05 or abs(rDer - rIzq) < 0.05:
        return 0.0      # diferencia del orden del ruido: superficie de frente
    y = (rDer * rDer - rIzq * rIzq) / (2.0 * sep)
    y = max(-r, min(r, y))
    b = math.atan2(y, math.sqrt(max(r * r - y * y, 1e-9)))
    lim = math.radians(US_MEDIO_CONO_DEG)
    return max(-lim, min(lim, b))


def rangoPiso(pitchDeg, altura=CAM_ALTURA_M, medioCono=US_MEDIO_CONO_DEG):
    """distancia a la que el borde de abajo del haz pega en el piso con
    la cabeza inclinada pitchDeg hacia abajo. None si el haz no toca
    piso dentro del alcance (cabeza derecha o viendo arriba)."""
    ang = math.radians(pitchDeg + medioCono)
    if ang <= math.radians(1.0):
        return None
    r = altura / math.sin(ang)
    return r if r < US_MAX_M else None


def esEcoDePiso(rango, pitchDeg, altura=CAM_ALTURA_M, margen=0.85):
    """True si la lectura cae donde el haz ya toca el piso: con la
    cabeza agachada los frontales rebotan en el suelo y eso NO es una
    pared. lo de antes de esa distancia si es obstaculo real."""
    if pitchDeg is None:
        return False
    rp = rangoPiso(pitchDeg, altura)
    return rp is not None and rango >= margen * rp
