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
    """los 2 MB1232 del frente (montaje.py: uno sobre cada lente,
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
    p.add_argument("--step-length", type=float, default=0.68,
                   help="longitud de paso (m) para el avance por pasos; "
                        "calibrar caminando una distancia conocida")
    args = p.parse_args()

    import cv2
    from vision_bridge import VisionBridge, GroundHazardDetector
    from odometry import makeMotionEstimator

    voz = VoiceOutput()
    voz.say("Iniciando sistema de navegacion", priority=2)

    bridge = VisionBridge(modelPath=args.model,
                          cameraProfile="picam3wide",
                          camHeightM=args.cam_height,
                          camPitchDeg=args.cam_pitch,
                          extraModelPath=None if args.sin_extra else "auto",
                          extraEvery=args.extra_every)
    detPeligros = GroundHazardDetector(
        cameraProfile="picam3wide", camHeightM=args.cam_height,
        camPitchDeg=args.cam_pitch) if args.hazards else None
    motionEst = makeMotionEstimator(
        useImu=not args.no_imu, imuBus=args.imu_bus,
        imuAddr=args.imu_addr, stepLengthM=args.step_length,
        hfovDeg=math.degrees(bridge.hfov),
        say=lambda m: voz.say(m, priority=2))
    ultrasonico = Mb1232Array()
    modAlertas = ModuloAlertas(
        hablarFn=lambda t: voz.say(t, priority=0))

    mgr = ChunkedMapManager(mapDir=args.map_dir,
                            chunkSizeM=args.chunk_size,
                            resolution=args.resolution)
    slam = EKFSlam(0.0, 0.0, 0.0)
    guardados = mgr.loadLandmarks()
    if guardados:
        slam.importLandmarks(guardados)
        mgr.marcarRelocPendiente()
        voz.say("Mapa conocido cargado", priority=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    lastT = time.time()
    lastSave= lastT
    ultimosPeligros = []
    frames = 0
    nav = None

    try:
        while args.max_frames is None or frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            frames += 1
            ahora = time.time()
            dt = max(1e-3, ahora - lastT)
            lastT =ahora

            motion, odo = motionEst.update(frame, dt)
            if odo["pitch_deg"] is not None:
                bridge.camPitchDeg = odo["pitch_deg"]
                if detPeligros is not None:
                    detPeligros.camPitchDeg = odo["pitch_deg"]
            detections = bridge.process(frame, egoYaw=motion.delta_rot)
            if detPeligros is not None and frames % args.hazard_every == 0:
                ultimosPeligros = detPeligros.process(frame, detections)
            detections = detections + ultimosPeligros

            # los frontales validan a la camara (y la imu quita el
            # rebote en el piso cuando agachas la cabeza)
            usLecturas = validarConUltrasonido(
                detections, ultrasonico.read(), odo["pitch_deg"])
            conectarUltrasonido(modAlertas, usLecturas,
                                pitchDeg=odo["pitch_deg"])
            observaciones = fuseObservations(detections, usLecturas)
            output = slam.step(motion, observaciones)
            pose = (output.x, output.y, output.theta)
            mgr.updateFromScan(pose, slam.observationsForMap(),
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

            alertas, _recalc = modAlertas.evaluar(nav)
            for prio, msg in alertas:
                print(f"[ALERTA p{prio}] {msg}")

            if ahora - lastSave > args.save_every:
                mgr.saveAll()
                mgr.saveLandmarks(slam.exportLandmarks())
                lastSave = ahora
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        n = mgr.saveAll()
        mgr.saveLandmarks(slam.exportLandmarks())
        print(f"\n[mapa] {n} chunks + landmarks guardados en {args.map_dir}")
        voz.say("Sistema detenido", priority=2)


if __name__ == "__main__":
    main()
