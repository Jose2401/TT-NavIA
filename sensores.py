"""
sensores.py — la imu y los 2 ultrasonicos de las gafas, que NO van al
i2c de la pi: los lee el RP2040-Zero de las gafas y los manda por usb
serie (firmware/code.py, lineas csv). aqui un hilo lee ese puerto a
50 Hz y deja todo listo pa los mains:

  - SensoresGafas: el hilo lector. cuenta pasos y acumula el giro A LA
    FRECUENCIA DE LA IMU (el ciclo de vision va a 2-5 Hz y a esa
    velocidad un paso no se alcanza a ver)
  - ImuGafas: misma interfaz que odometry.Bno055 + tomarMovimiento()
  - UltrasonicosGafas: mismo read() que main_pi.Mb1232Array

el giro sale del heading fusionado del bno055, pero vigilado con el
giroscopio: en modo NDOF el chip a veces "brinca" el rumbo cuando el
magnetometro se calibra o pasa junto a algo de fierro; si el heading
cambia y el giroscopio dice que la cabeza no giro, ese brinco se
ignora (se usa el giro del giroscopio).
"""
import collections
import math
import threading
import time

import montaje
from interfaces import UltrasonicReading
from odometry import MontajeImu, StepDetector

VID_RP2040 = 0x2E8A
SIN_ECO_CM = 765
_MASK_MS = (1 << 29) - 1          # supervisor.ticks_ms() da la vuelta ahi

# heading del bno055 vs giroscopio: si difieren mas que esto en una
# muestra (20 ms), el heading brinco solo
BRINCO_DEG = 1.5
VIEJA_S = 0.6                     # lectura de ultrasonido caducada
SILENCIO_S = 3.0                  # sin lineas = el firmware se cayo

# contador de pasos con la aceleracion VERTICAL en la cabeza. ajustado
# con una caminata real (2026-11-10, pasillo, 10 pasos a 75 pasos/min):
# cada paso da un pico de 1.3 a 2.9 m/s2 y un valle de -0.7 a -1.5;
# parado y volteando no pasa de 0.9. con 1.8/0.8 solo contaba 5 de 10;
# de 1.0 a 1.2 cuenta los 10
PASO_UMBRAL = 1.1
PASO_REARME = 0.4
PASO_HUECO_MAX_S = 1.4            # mas que esto entre picos = ya no camina


def detectorPasosGafas():
    return StepDetector(threshold=PASO_UMBRAL, resetLevel=PASO_REARME,
                        confirmar=True, maxGapS=PASO_HUECO_MAX_S)


def buscarPuerto():
    """el /dev/ttyACM* del rp2040 de las gafas (por su id usb)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    puertos = list(list_ports.comports())
    for p in puertos:
        if p.vid == VID_RP2040:
            return p.device
    for p in puertos:
        if "ACM" in p.device:
            return p.device
    return None


def _wrapDeg(a):
    return (a + 180.0) % 360.0 - 180.0


class SensoresGafas:
    def __init__(self, puerto=None, alEco=None, iniciar=True,
                 stepDetector=None):
        """alEco(lecturas): se llama desde el hilo cada que llega un eco
        nuevo (pa pitar al momento, sin esperar a la camara)."""
        self.puerto = puerto
        self.alEco = alEco
        self.alLinea = None           # callable(linea, t): pa grabar sesiones
        self.razon = ""
        self._ser = None
        self._lock = threading.Lock()
        self._parar = False
        self.pasos = stepDetector or detectorPasosGafas()
        self.metros = 0.0             # caminado en total (pasos x cadencia)
        self.avanceM = 0.0            # lo del ultimo tomarMovimiento()
        self.alPaso = None            # callable(t, n, metros): pa analisis
        self.tPaso = 0.0              # cuando se conto el ultimo paso
        # lo de esta corrida, pa medir al final el largo de paso real
        # con el ultrasonido (aprenderPaso). ~1 h de datos
        self._regEco = collections.deque(maxlen=40000)
        self._regPasos = collections.deque(maxlen=20000)

        # imu (crudo, convencion del bno055 ya remapeado a las gafas)
        self.euler = None             # (heading, roll, pitch) en grados
        self.giro = (0.0, 0.0, 0.0)   # grados/s
        self.acelLineal = (0.0, 0.0, 0.0)
        self.gravedad = (0.0, 0.0, 9.8)
        self.acelVertical = 0.0
        self.calib = None
        self.tImu = 0.0               # time.time() de la ultima muestra
        self.muestrasImu = 0
        self.brincos = 0
        self._msPrev = None
        self._tMuestra = 0.0          # reloj del firmware, en s
        self._giroAcumDeg = 0.0       # crudo (sentido del bno055)
        self._pasosAcum = 0
        self._metrosAcum = 0.0
        self._signoGiro = -1.0        # heading del bno sube a la derecha
        self._corrSigno = 0.0

        # ultrasonicos: lado -> (cm, time.time())
        self._eco = {}
        self.version = None
        self.errores = []

        if iniciar:
            self.iniciar()

    # ---- conexion ----
    def iniciar(self):
        try:
            import serial  # noqa: F401
        except ImportError:
            self.razon = "pyserial no instalado"
            return False
        if not self._abrir():
            return False
        threading.Thread(target=self._loop, daemon=True).start()
        # un ratito pa que lleguen las primeras lineas
        t0 = time.time()
        while time.time() - t0 < 1.5 and not (self.imuOk and self.usOk):
            time.sleep(0.02)
        if not self.imuOk and not self.usOk:
            self.razon = (f"{self.puerto} abierto pero sin datos "
                          "(¿firmware/code.py cargado en el rp2040?)")
        return True

    def _abrir(self):
        import serial
        puerto = self.puerto or buscarPuerto()
        if puerto is None:
            self.razon = "no se encontró el RP2040 (¿cable usb de datos?)"
            return False
        try:
            self._ser = serial.Serial(puerto, 115200, timeout=0.5)
            self._ser.reset_input_buffer()
        except (serial.SerialException, OSError) as e:
            self.razon = f"no se pudo abrir {puerto} ({e})"
            self._ser = None
            return False
        self.puerto = puerto
        self.razon = ""
        return True

    def cerrar(self):
        self._parar = True
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass

    def _loop(self):
        import serial
        ultimaLinea = time.time()
        while not self._parar:
            if self._ser is None:
                time.sleep(2.0)
                if self._abrir():
                    ultimaLinea = time.time()
                continue
            try:
                cruda = self._ser.readline()
            except (serial.SerialException, OSError, TypeError):
                self.razon = "se desconectó el RP2040"
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                continue
            ahora = time.time()
            if cruda:
                ultimaLinea = ahora
                linea = cruda.decode("utf-8", "replace")
                self.procesarLinea(linea, ahora)
                if self.alLinea is not None:
                    self.alLinea(linea, ahora)
            elif ahora - ultimaLinea > SILENCIO_S:
                # el code.py se detuvo (error o alguien mando ctrl-c):
                # ctrl-d lo vuelve a arrancar
                try:
                    self._ser.write(b"\x04")
                except Exception:
                    pass
                ultimaLinea = ahora

    # ---- parseo (sin hardware: se prueba con lineas de texto) ----
    def procesarLinea(self, linea, ahora=None):
        ahora = time.time() if ahora is None else ahora
        if "\x1b" in linea:
            # circuitpython mete el titulo de la terminal entre escapes
            linea = linea.rsplit("\x1b\\", 1)[-1]
        partes = linea.strip().split(",")
        tipo = partes[0]
        try:
            if tipo == "I" and len(partes) == 15:
                self._muestraImu([int(x) for x in partes[1:]], ahora)
            elif tipo == "U" and len(partes) == 4:
                self._muestraEco(partes[2], int(partes[3]), ahora)
            elif tipo == "H" and len(partes) >= 3:
                self.version = int(partes[2])
            elif tipo == "E":
                self.errores.append((ahora, ",".join(partes[1:])))
                del self.errores[:-20]
        except ValueError:
            pass

    def _muestraImu(self, v, ahora):
        ms, h, r, p, wx, wy, wz, lx, ly, lz, gx, gy, gz, cal = v
        heading, roll, pitch = h / 16.0, r / 16.0, p / 16.0
        giro = (wx / 16.0, wy / 16.0, wz / 16.0)
        lin = (lx / 100.0, ly / 100.0, lz / 100.0)
        grav = (gx / 100.0, gy / 100.0, gz / 100.0)
        gMag = math.sqrt(grav[0] ** 2 + grav[1] ** 2 + grav[2] ** 2)

        dt = 0.0
        if self._msPrev is not None:
            dt = ((ms - self._msPrev) & _MASK_MS) / 1000.0
            if dt > 1.0:
                dt = 0.0              # hueco (reinicio del firmware)
        self._msPrev = ms
        self._tMuestra += dt

        # giro alrededor de la vertical segun el giroscopio, y componente
        # vertical de la aceleracion (+ = hacia arriba)
        if gMag > 1.0:
            wVert = (giro[0] * grav[0] + giro[1] * grav[1]
                     + giro[2] * grav[2]) / gMag
            aVert = (lin[0] * grav[0] + lin[1] * grav[1]
                     + lin[2] * grav[2]) / gMag
        else:
            wVert = giro[2]
            aVert = math.sqrt(lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2)

        dGiro = 0.0
        brinco = False
        if self.euler is not None and dt > 0.0:
            dHead = _wrapDeg(heading - self.euler[0])
            dGyro = wVert * dt
            if abs(dHead) - abs(dGyro) > BRINCO_DEG:
                brinco = True
                dGiro = self._signoGiro * dGyro
            else:
                dGiro = dHead
                # se aprende de que lado gira el heading respecto al
                # giroscopio (por si el montaje lo invierte)
                self._corrSigno = 0.999 * self._corrSigno + dHead * dGyro
                if abs(self._corrSigno) > 5.0:
                    self._signoGiro = 1.0 if self._corrSigno > 0 else -1.0

        nPasos = int(self.pasos.update(aVert, self._tMuestra))
        # cada paso avanza segun la cadencia a la que va (montaje.largoPaso)
        avance = nPasos * montaje.largoPaso(self.pasos.cadencia)
        if nPasos:
            self.tPaso = ahora
            self._regPasos.extend([ahora] * nPasos)
            if self.alPaso is not None:
                self.alPaso(ahora, nPasos, avance)
        with self._lock:
            self.euler = (heading, roll, pitch)
            self.giro, self.acelLineal, self.gravedad = giro, lin, grav
            self.acelVertical = aVert
            self.calib = {"sys": (cal >> 6) & 3, "gyro": (cal >> 4) & 3,
                          "accel": (cal >> 2) & 3, "mag": cal & 3}
            self.tImu = ahora
            self.muestrasImu += 1
            self._giroAcumDeg += dGiro
            self._pasosAcum += nPasos
            self._metrosAcum += avance
            self.metros += avance
            if brinco:
                self.brincos += 1

    def _muestraEco(self, lado, cm, ahora):
        lado = {"I": "izq", "D": "der"}.get(lado)
        if lado is None:
            return
        with self._lock:
            if cm < 0:
                self._eco.pop(lado, None)      # no responde en el bus
            else:
                self._eco[lado] = (cm, ahora)
                self._regEco.append((ahora, lado, cm))
        if self.alEco is not None and cm >= 0:
            try:
                self.alEco(self.lecturasUS(ahora))
            except Exception as e:
                print(f"[SENSORES] alEco fallo: {e}")

    # ---- consulta ----
    @property
    def imuOk(self):
        return self.euler is not None and time.time() - self.tImu < 1.0

    @property
    def usOk(self):
        ahora = time.time()
        with self._lock:
            return any(ahora - t < 1.5 for _cm, t in self._eco.values())

    @property
    def available(self):
        return self.imuOk or self.usOk

    def caminando(self, ahora=None, margenS=2.5):
        """True si se conto un paso hace poco (las alertas de cosas que
        no se mueven solo se repiten si el usuario va caminando)."""
        ahora = time.time() if ahora is None else ahora
        return ahora - self.tPaso < margenS

    def pasoMedido(self):
        """el largo de paso REAL de esta corrida segun el ultrasonido
        (ecoContraPasos) o None si no hubo donde medirlo."""
        return ecoContraPasos(list(self._regEco), list(self._regPasos))

    def aprenderPaso(self, minMetros=5.0, peso=0.3, guardar=True):
        """al terminar una corrida: si el ultrasonido vio acercarse algo
        al menos minMetros en total mientras se caminaba, afina el largo
        de paso de quien trae los lentes y lo guarda pa la siguiente
        (calibracion_usuario.json). asi nadie tiene que medir 10 m.
        -> (nuevo, medido) o None si no hubo con que."""
        e = self.pasoMedido()
        if e is None or e["metros"] < minMetros or e["pasos"] < 8:
            return None
        medido = e["largoPasoM"]
        if not 0.25 <= medido <= 1.0:
            return None
        nuevo = medido
        if montaje.PASO_CALIBRADO:
            # ya habia uno: se mueve tantito hacia lo medido hoy
            nuevo = (1.0 - peso) * montaje.LARGO_PASO_M + peso * medido
        nuevo = round(nuevo, 3)
        if guardar:
            montaje.guardarUsuario(largoPasoM=nuevo, pasoFuente="ultrasonido",
                                   pasoFecha=time.strftime("%Y-%m-%d"))
        return nuevo, medido

    def tomarMovimiento(self):
        """-> (giroCrudoDeg, pasos) acumulados desde la ultima llamada.
        lo que avanzo en esos pasos (en metros) queda en self.avanceM."""
        with self._lock:
            g, n = self._giroAcumDeg, self._pasosAcum
            self.avanceM = self._metrosAcum
            self._giroAcumDeg, self._pasosAcum = 0.0, 0
            self._metrosAcum = 0.0
        return g, n

    def lecturasUS(self, ahora=None):
        from montaje import usPosicion, US_MAX_M
        ahora = time.time() if ahora is None else ahora
        with self._lock:
            eco = dict(self._eco)
        out = []
        for lado in ("izq", "der"):
            if lado not in eco or ahora - eco[lado][1] > VIEJA_S:
                continue
            cm = eco[lado][0]
            oy, b = usPosicion(lado)
            # 765 = no hubo eco. ojo: 20 cm NO es invalido, el sensor
            # da 20 pa todo lo que este a 20 cm o MAS CERCA
            out.append(UltrasonicReading(
                range_m=cm / 100.0, sensor_bearing=b, max_range=US_MAX_M,
                valid=cm < SIN_ECO_CM - 5, offset_y=oy, lado=lado))
        return out


class ImuGafas(MontajeImu):
    """la imu de las gafas vista como la ve odometry.MotionEstimator.
    el firmware ya remapea los ejes (imu de pie en el puente): pitch +
    = frente hacia el piso, heading sube al girar a la derecha."""

    def __init__(self, sensores, montajePath=None):
        self.s = sensores
        self.placement = None
        self.avanceM = 0.0            # metros del ultimo tomarMovimiento()
        self._cargarMontaje(montajePath or self.MONTAJE_JSON)

    @property
    def available(self):
        return self.s.imuOk

    @property
    def reason(self):
        return self.s.razon or "el RP2040 no manda datos de la imu"

    def readEulerCrudo(self):
        return self.s.euler if self.s.imuOk else None

    def readLinearAccel(self):
        return self.s.acelLineal if self.s.imuOk else None

    def calibration(self):
        return self.s.calib

    def tomarMovimiento(self):
        """-> (giro en rad ya en la convencion del sistema (+ = izq),
        pasos nuevos, pitch en grados (+ = abajo))"""
        euler = self.readEuler()
        if euler is None:
            return None
        giroDeg, pasos = self.s.tomarMovimiento()
        self.avanceM = self.s.avanceM
        return (math.radians(self.HEADING_SIGN * giroDeg), pasos,
                math.degrees(euler[2]))


class UltrasonicosGafas:
    MAX_RANGE_M = 7.65

    def __init__(self, sensores):
        self.s = sensores

    @property
    def available(self):
        return self.s.usOk

    @property
    def razon(self):
        return self.s.razon or "el RP2040 no manda ecos"

    def read(self):
        return self.s.lecturasUS()


def avancePorEco(ecos, pasosT=None):
    """cuanto se avanzo segun los ultrasonicos: en los tramos en que hay
    algo enfrente y la lectura va BAJANDO pareja, lo que baja es lo que
    se camino (medida directa, no depende del largo de paso).
    ecos: [(t, lado, cm)]. pasosT: tiempos de los pasos; si se dan, solo
    cuentan los tramos caminando (que no sea alguien acercandose o la
    cabeza girando hacia una pared).
    -> {"metros", "segundos", "tramos": [(t0, t1)]}"""
    PASO_T = 0.02
    vel = {}                          # casilla de tiempo -> [m/s por sensor]
    for lado in ("izq", "der", "I", "D"):
        serie = [(t, cm) for t, l, cm in ecos if l == lado]
        corrida = []                  # [(t, cm)] bajando parejo
        def cerrar(corrida):
            if len(corrida) < 4 or corrida[0][1] - corrida[-1][1] < 30:
                return
            for (ta, ca), (tb, cb) in zip(corrida, corrida[1:]):
                v = (ca - cb) / 100.0 / (tb - ta)
                k = int(ta / PASO_T)
                while k * PASO_T < tb:
                    vel.setdefault(k, []).append(v)
                    k += 1
        for t, cm in serie:
            ok = 20 < cm < SIN_ECO_CM - 5
            if ok and corrida:
                dt = t - corrida[-1][0]
                baja = corrida[-1][1] - cm
                # entre 0.05 y 2.2 m/s acercandose
                if not (0.0 < dt <= 0.45 and 5.0 * dt <= baja <= 220.0 * dt):
                    cerrar(corrida)
                    corrida = []
            if ok:
                corrida.append((t, cm))
            else:
                cerrar(corrida)
                corrida = []
        cerrar(corrida)
    if pasosT is not None:
        vel = {k: v for k, v in vel.items()
               if any(abs(k * PASO_T - tp) <= 1.0 for tp in pasosT)}
    metros = sum(sum(v) / len(v) for v in vel.values()) * PASO_T
    tramos = []
    for k in sorted(vel):
        t = k * PASO_T
        if tramos and t - tramos[-1][1] <= 2 * PASO_T:
            tramos[-1][1] = t + PASO_T
        else:
            tramos.append([t, t + PASO_T])
    return {"metros": metros, "segundos": len(vel) * PASO_T,
            "tramos": [(round(a, 2), round(b, 2)) for a, b in tramos]}


def ecoContraPasos(ecos, tPasos):
    """compara lo que avanzo segun el ultrasonido con los pasos que se
    CONTARON en esos mismos tramos. ecos: [(t, lado, cm)], tPasos: el
    tiempo de cada paso. -> dict o None si no hubo tramos.
    largoPasoM = metros / pasos contados: es justo lo que la odometria
    necesita (si el detector se come un paso, el largo lo compensa)."""
    eco = avancePorEco(ecos, tPasos)
    if eco["segundos"] < 1.0:
        return None
    pasos = sum(1 for tp in tPasos
                if any(a <= tp <= b for a, b in eco["tramos"]))
    if pasos < 2:
        return None
    return {"metros": round(eco["metros"], 2),
            "segundos": round(eco["segundos"], 1),
            "velocidad": round(eco["metros"] / eco["segundos"], 2),
            "pasos": pasos,
            "largoPasoM": round(eco["metros"] / pasos, 3),
            "tramos": eco["tramos"]}


def resumenCaminata(ruta):
    """pasa una grabacion cruda (lineas "t linea", como serie.txt de
    grabar_sesion.py) por el MISMO contador de pasos y dice cuanto se
    camino: por pasos, y de comprobacion por los ultrasonicos
    (ecoContraPasos) donde se pudo."""
    s = SensoresGafas(iniciar=False)
    pasos = []                        # (t, n, metros)
    s.alPaso = lambda t, n, m: pasos.append((t, n, m))
    t = t0 = None
    with open(ruta, encoding="utf-8") as fh:
        for ln in fh:
            partes = ln.split(None, 1)
            if len(partes) < 2:
                continue
            try:
                t = float(partes[0])
            except ValueError:
                continue
            t0 = t if t0 is None else t0
            s.procesarLinea(partes[1], t)
    n = sum(p[1] for p in pasos)
    tPasos = [p[0] for p in pasos]
    huecos = sorted(b - a for a, b in zip(tPasos, tPasos[1:])
                    if b - a <= PASO_HUECO_MAX_S)
    cadencia = 60.0 / huecos[len(huecos) // 2] if huecos else None
    out = {"pasos": n, "metros": round(s.metros, 2),
           "cadenciaPpm": round(cadencia, 1) if cadencia else None,
           "duracionS": round((t - t0), 1) if t0 is not None else 0.0,
           "eco": s.pasoMedido()}
    if out["eco"] and n:
        # que tanto difiere el paso real del que se uso en esta cuenta
        out["eco"]["factor"] = round(out["eco"]["largoPasoM"]
                                     / (s.metros / n), 3)
    return out


def textoResumen(r):
    """el resumen de arriba en un renglon pa imprimir."""
    txt = (f"{r['pasos']} pasos, {r['metros']:.1f} m"
           + (f" ({r['cadenciaPpm']:.0f} pasos/min)" if r["cadenciaPpm"]
              else ""))
    e = r.get("eco")
    if e:
        txt += (f" | ultrasonido: {e['metros']:.1f} m en {e['segundos']:.1f} s "
                f"con algo enfrente ({e['velocidad']:.2f} m/s, {e['pasos']} "
                f"pasos) -> paso real de {e['largoPasoM']:.2f} m"
                + (f", {e['factor']:.2f} veces el que se usó"
                   if "factor" in e else ""))
    return txt


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # python sensores.py sesiones/x/serie.txt -> cuanto se camino
        print(textoResumen(resumenCaminata(sys.argv[1])))
        raise SystemExit
    # monitor en vivo: python sensores.py
    s = SensoresGafas()
    if not s.available:
        raise SystemExit(f"sensores no disponibles: {s.razon}")
    imu = ImuGafas(s)
    print(f"puerto {s.puerto} | firmware v{s.version} | montaje imu: "
          f"{imu.montajeDesde}")
    pasosTot = 0
    try:
        while True:
            time.sleep(0.2)
            e = imu.readEuler()
            _g, n = s.tomarMovimiento()
            pasosTot += n
            ecos = {u.lado: (f"{u.range_m:.2f} m" if u.valid else "sin eco")
                    for u in s.lecturasUS()}
            if e is None:
                print("imu sin datos", end="\r")
                continue
            c = s.calib
            print("heading %+7.1f  pitch %+6.1f (+ = abajo)  pasos %3d  "
                  "izq %-8s der %-8s  cal S%d G%d A%d M%d   "
                  % (math.degrees(e[0]), math.degrees(e[2]), pasosTot,
                     ecos.get("izq", "ausente"), ecos.get("der", "ausente"),
                     c["sys"], c["gyro"], c["accel"], c["mag"]), end="\r")
    except KeyboardInterrupt:
        print()
