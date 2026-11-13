import argparse
import math
import os
import threading
import time

from interfaces import UltrasonicReading

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from vision_bridge import fuseObservations, validarConUltrasonido
from navigation_feed import buildNavigationFrame
from alertas import ModuloAlertas, conectarUltrasonido


class Mb1232Array:
    """RESPALDO: MB1232 conectados DIRECTO al i2c de la pi (el diseño
    original). en el prototipo final los lee el rp2040 de las gafas y
    llegan por usb: ver sensores.UltrasonicosGafas / abrirGafas().

    los 2 MB1232 del frente (montaje.py: uno sobre cada lente,
    apuntando AL FRENTE, no a los lados). como ven lo mismo, si
    disparan juntos uno oye el eco del otro: se disparan ALTERNADOS en
    un hilo (~80 ms cada uno -> ~6 Hz por sensor) y read() regresa lo
    ultimo sin bloquear el loop de la camara. protocolo i2cxl: escribir
    0x51 = medir, esperar, leer 2 bytes = cm. direcciones segun el
    diseño (cambiarlas con el comando de re-direccion del fabricante)."""
    MAX_RANGE_M = 7.65
    CMD_MEDIR = 0x51
    # datasheet: sin monitorear el pin 2 (status) hay que dejar 100 ms
    # entre ordenar la medicion y leer
    ESPERA_S = 0.10
    VIEJA_S = 0.5
    # direccion de fabrica: 224 (8 bits) = 0x70 en 7 bits. pa tener dos
    # en el mismo bus hay que re-direccionar uno con la secuencia del
    # fabricante (0xAA, 0xA5, nueva_dir) — el diseño usa 0x20/0x21

    def __init__(self, addrLeft=0x20, addrRight=0x21, bus=1):
        from montaje import usPosicion
        self.sensores = {"izq": (addrLeft, *usPosicion("izq")),
                         "der": (addrRight, *usPosicion("der"))}
        self.available = False
        self.razon = ""
        self._ultimo = {}
        self._lock = threading.Lock()
        try:
            from smbus2 import SMBus
            self._bus = SMBus(bus)
            for addr, _oy, _b in self.sensores.values():
                self._bus.write_byte(addr, self.CMD_MEDIR)
            self.available = True
            threading.Thread(target=self._loop, daemon=True).start()
        except ImportError:
            self.razon = "smbus2 no instalado"
        except OSError as e:
            self.razon = f"sin respuesta i2c ({e})"

    def _medir(self, addr):
        self._bus.write_byte(addr, self.CMD_MEDIR)
        time.sleep(self.ESPERA_S)
        alto, bajo = self._bus.read_i2c_block_data(addr, 0, 2)
        return ((alto << 8) | bajo) / 100.0

    def _loop(self):
        while True:
            for lado, (addr, _oy, _b) in self.sensores.items():
                try:
                    r = self._medir(addr)
                except OSError:
                    continue
                with self._lock:
                    self._ultimo[lado] = (r, time.time())

    def read(self):
        if not self.available:
            return []
        ahora = time.time()
        out = []
        with self._lock:
            ultimo = dict(self._ultimo)
        for lado, (addr, oy, b) in self.sensores.items():
            if lado not in ultimo or ahora - ultimo[lado][1] > self.VIEJA_S:
                continue
            r = ultimo[lado][0]
            # 7.65 = no hubo eco. OJO: 20 cm NO es invalido, el sensor
            # reporta 20 pa todo lo que este a 20 cm o MAS CERCA (lo mas
            # peligroso); se pasa tal cual pa que pite critico
            valido = r < self.MAX_RANGE_M - 0.05
            out.append(UltrasonicReading(
                range_m=r, sensor_bearing=b, max_range=self.MAX_RANGE_M,
                valid=valido, offset_y=oy, lado=lado))
        return out


def abrirGafas(puerto=None, usar=True):
    """-> (sensores, ultrasonico). sensores = el hilo que lee el rp2040
    de las gafas (None si no esta); ultrasonico = lo que tenga read():
    los de las gafas o, de respaldo, los del i2c de la pi."""
    sensores = None
    if usar:
        try:
            from sensores import SensoresGafas, UltrasonicosGafas
            sensores = SensoresGafas(puerto=puerto)
            if sensores.available:
                print(f"[GAFAS] rp2040 en {sensores.puerto}: imu "
                      f"{'ok' if sensores.imuOk else 'NO'}, ultrasónicos "
                      f"{'ok' if sensores.usOk else 'NO'}")
                return sensores, UltrasonicosGafas(sensores)
            print(f"[GAFAS] sin sensores por usb ({sensores.razon})")
            sensores.cerrar()
            sensores = None
        except Exception as e:
            print(f"[GAFAS] sin sensores por usb ({e})")
            sensores = None
    us = Mb1232Array()
    if not us.available:
        print(f"[GAFAS] sin ultrasónicos ({us.razon})")
    return sensores, us


def engancharEcos(sensores, modAlertas, pitchFn):
    """los ecos pitan desde el hilo del sensor (10 por segundo), sin
    esperar al ciclo de la camara. pitchFn() -> cabeceo en grados o
    None (pa descartar el rebote en el piso)."""
    if sensores is None:
        return False

    def alEco(lecturas):
        conectarUltrasonido(modAlertas, lecturas, pitchDeg=pitchFn())
    sensores.alEco = alEco
    # parado junto a algo quieto la alerta se dice una vez; se repite
    # cuando la imu vuelve a contar pasos
    modAlertas.caminandoFn = sensores.caminando
    return True


def _alDisco():
    """lo recien guardado (mapa, calibracion) de verdad a la tarjeta, en
    otro hilo pa no frenar el ciclo: sin monitor la pi se apaga
    desconectandola y lo de los ultimos ~30 s se perdia."""
    threading.Thread(target=os.sync, daemon=True).start()


def _aprenderPaso(sensores):
    """al salir: si en esta corrida el ultrasonido pudo medir el avance,
    se afina el largo de paso de quien trae los lentes."""
    try:
        r = sensores.aprenderPaso()
    except Exception as e:
        print(f"[PASO] no se pudo afinar ({e})")
        return
    if r is not None:
        print(f"[PASO] el ultrasonido midió pasos de {r[1]:.2f} m en esta "
              f"corrida; queda en {r[0]:.2f} m pa la siguiente")


class Cronometro:
    """tiempos por etapa del ciclo (pa --estadisticas y el documento)."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.t0 = time.time()
        self.n = 0
        self.suma = {}
        self._marca = None

    def inicio(self):
        self._marca = time.time()

    def etapa(self, nombre):
        ahora = time.time()
        self.suma[nombre] = self.suma.get(nombre, 0.0) + ahora - self._marca
        self._marca = ahora

    def cuadro(self):
        self.n += 1

    def resumen(self):
        dur = max(1e-6, time.time() - self.t0)
        n = max(1, self.n)
        partes = "  ".join(f"{k} {1000 * v / n:.0f}ms"
                           for k, v in self.suma.items())
        return f"{self.n / dur:.1f} cuadros/s | {partes} | {_recursos()}"


def _recursos():
    """memoria del proceso y temperatura del cpu (solo linux/pi)."""
    out = []
    try:
        with open("/proc/self/statm") as fh:
            paginas = int(fh.read().split()[1])
        out.append(f"ram {paginas * os.sysconf('SC_PAGE_SIZE') / 1e6:.0f}MB")
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            out.append(f"cpu {int(fh.read()) / 1000:.0f}C")
    except (OSError, ValueError):
        pass
    return " ".join(out)


class VoiceOutput:
    def __init__(self):
        self._ultimo = {}

    # no repite la misma frase en rafaga
    def say(self, msg, priority=2, repeatAfterS=4.0):
        ahora = time.time()
        if ahora - self._ultimo.get(msg, 0.0) < repeatAfterS:
            return
        self._ultimo[msg] = ahora
        print(f"[VOZ p{priority}] {msg}", flush=True)


# loop headless de la pi
def main():
    p = argparse.ArgumentParser(description="Main Raspberry Pi (headless)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=None, help="pesos YOLOv8-seg")
    p.add_argument("--sin-extra", action="store_true",
                   help="apagar el detector extra (puertas/escaleras, oiv7)")
    p.add_argument("--extra-every", type=int, default=15)
    p.add_argument("--cam-height", type=float, default=1.55)
    p.add_argument("--cam-pitch", type=float, default=10.0)
    p.add_argument("--hazards", action="store_true",
                   help="peligros de piso (SegFormer; costoso en la Pi)")
    p.add_argument("--hazard-every", type=int, default=10)
    p.add_argument("--map-dir", default=os.path.join("maps", "default"))
    p.add_argument("--fresh", action="store_true",
                   help="borrar el mapa guardado y empezar de cero")
    p.add_argument("--chunk-size", type=float, default=8.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-range", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--save-every", type=float, default=30.0,
                   help="autoguardado del mapa cada N segundos")
    p.add_argument("--no-imu", action="store_true",
                   help="no usar la IMU BNO055 (solo flujo óptico)")
    p.add_argument("--imu-addr", type=lambda s: int(s, 0), default=0x28,
                   help="dirección I2C del BNO055 (0x28 o 0x29)")
    p.add_argument("--imu-bus", type=int, default=1)
    p.add_argument("--step-length", type=float, default=None,
                   help="longitud de paso (m) para el avance por pasos; "
                        "calibrar caminando una distancia conocida")
    p.add_argument("--puerto", default=None,
                   help="puerto serie del RP2040 de las gafas (auto)")
    p.add_argument("--sin-gafas", action="store_true",
                   help="no buscar el RP2040 (sensores por i2c o ninguno)")
    p.add_argument("--umbral-alerta", type=float, default=None,
                   help="metros a los que suena la alerta hablada de un "
                        "obstáculo fijo (por defecto 0.25)")
    p.add_argument("--umbral-precaucion", type=float, default=None,
                   help="metros a los que empieza el pitido de precaución "
                        "(por defecto 0.60)")
    p.add_argument("--estadisticas", type=float, default=0.0,
                   help="imprimir cuadros/s y tiempos cada N segundos")
    args = p.parse_args()

    from camara import abrirCamara
    from vision_bridge import (VisionBridge, GroundHazardDetector,
                               freeSpaceRays)
    from odometry import makeMotionEstimator

    voz = VoiceOutput()
    voz.say("Iniciando sistema de navegacion", priority=2)

    cap, perfil = abrirCamara(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara "
                         f"({getattr(cap, 'razon', '')})")
    sensores, ultrasonico = abrirGafas(args.puerto, usar=not args.sin_gafas)

    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile=perfil,
                          camHeightM=args.cam_height,
                          camPitchDeg=args.cam_pitch,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    detPeligros = GroundHazardDetector(
        cameraProfile=perfil, camHeightM=args.cam_height,
        camPitchDeg=args.cam_pitch) if args.hazards else None
    motionEst = makeMotionEstimator(
        useImu=not args.no_imu, imuBus=args.imu_bus,
        imuAddr=args.imu_addr, stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov),
        say=lambda m: voz.say(m, priority=2), sensores=sensores)
    modAlertas = ModuloAlertas(
        hablarFn=lambda t: voz.say(t, priority=0),
        criticaM=args.umbral_alerta, precaucionM=args.umbral_precaucion)
    pitchActual = [None]
    ecosEnHilo = engancharEcos(sensores, modAlertas, lambda: pitchActual[0])

    if args.fresh and os.path.isdir(args.map_dir):
        import shutil
        shutil.rmtree(args.map_dir)
        print(f"[mapa] {args.map_dir} borrado (--fresh)")
    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        mgr.marcarRelocPendiente()
        voz.say("Mapa conocido cargado", priority=2)

    lastT = time.time()
    lastSave= lastT
    ultimosPeligros = []
    frames = 0
    nav = None
    crono = Cronometro()

    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT =ahora
            crono.inicio()

            motion, odo = motionEst.update(frame, dt)
            pitchActual[0] = odo["pitch_deg"]
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
                if detPeligros is not None:
                    detPeligros.camPitchDeg = odo["pitch_deg"]
            crono.etapa("odometria")
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and frames % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros
            crono.etapa("vision")

            # los frontales validan a la camara (y la imu quita el
            # rebote en el piso cuando agachas la cabeza)
            usLecturas = validarConUltrasonido(
                detections, ultrasonico.read(), odo["pitch_deg"])
            if not ecosEnHilo:
                conectarUltrasonido(modAlertas, usLecturas,
                                    pitchDeg=odo["pitch_deg"])
            observaciones = fuseObservations(detections, usLecturas)
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            crono.etapa("slam")
            mgr.updateFromScan(pose, slam.observationsForMap()
                               + freeSpaceRays(usLecturas, maxLibreM=2.0)
                               + (detPeligros.tomarPisoLibre() if detPeligros else []),
                           maxRange=args.max_range)
            mgr.clearFootprint(output.x, output.y)
            eventos = mgr.updatePosition(pose, slam.getLandmarks())
            if mgr.checarReloc(slam):
                voz.say("Ubicacion confirmada en el mapa conocido.", priority=2)

            for ev in eventos:
                if ev["type"] == "chunk_change" and not ev["known"]:
                    voz.say("Zona nueva", priority=2)
                elif ev["type"] == "room_recognized":
                    voz.say(f"Estas en {ev['room']}", priority=1)
                elif ev["type"] == "door_crossed":
                    voz.say("Cruzaste una puerta", priority=1)

            nav = buildNavigationFrame(
                slam, mgr, detections, maxRange=args.max_range,
                chunk=mgr.currentChunk, room=mgr.currentRoom,
                ultrasonic=usLecturas)

            crono.etapa("mapa")
            alertas, _recalc = modAlertas.evaluar(nav)
            for prio, msg in alertas:
                print(f"[ALERTA p{prio}] {msg}")
            crono.etapa("alertas")
            crono.cuadro()
            if args.estadisticas and time.time() - crono.t0 > args.estadisticas:
                print(f"[CICLO] {crono.resumen()}", flush=True)
                crono.reiniciar()

            if ahora - lastSave > args.save_every:
                mgr.saveAll()
                mgr.saveLandmarks(slam.exportLandmarks())
                _alDisco()
                lastSave = ahora
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if sensores is not None:
            _aprenderPaso(sensores)
            sensores.cerrar()
        n = mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        _alDisco()
        print(f"\n[mapa] {n} chunks + landmarks guardados en {args.map_dir}")
        voz.say("Sistema detenido", priority=2)


if __name__ == "__main__":
    main()
