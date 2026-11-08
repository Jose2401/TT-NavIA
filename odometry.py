import math
import time

from interfaces import MotionEstimate


def _wrapAng(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Bno055:
    CHIP_ID = 0xA0
    REG_CHIP_ID = 0x00
    REG_OPR_MODE = 0x3D
    REG_PWR_MODE = 0x3E
    REG_SYS_TRIGGER = 0x3F
    REG_UNIT_SEL = 0x3B
    REG_EULER_H = 0x1A
    REG_LIN_ACCEL = 0x28
    REG_CALIB_STAT = 0x35
    MODE_CONFIG = 0x00
    MODE_NDOF = 0x0C

    REG_AXIS_MAP_CONFIG = 0x41
    REG_AXIS_MAP_SIGN = 0x42
    # placements P0-P7 del datasheet del bno055 (config, signo)
    PLACEMENTS = {"P0": (0x21, 0x04), "P1": (0x24, 0x00),
                  "P2": (0x24, 0x06), "P3": (0x21, 0x02),
                  "P4": (0x24, 0x03), "P5": (0x21, 0x01),
                  "P6": (0x21, 0x07), "P7": (0x24, 0x05)}
    MONTAJE_JSON = "imu_montaje.json"

    # va plana arriba al centro del armazon (montaje.py). cual campo es
    # el cabeceo y con que signo depende de como se atornille la placa:
    # eso lo mide `python odometry.py --calibrar` y lo guarda en el json
    HEADING_SIGN = -1.0
    PITCH_SIGN = 1.0
    PITCH_DESDE = "pitch"        # "pitch" o "roll" del bno055
    PITCH_OFFSET_DEG = 0.0       # lectura con la cabeza derecha = 0

    def __init__(self, bus=1, address=0x28, placement=None,
                 montajePath=None):
        self.address = address
        self.available= False
        self.reason = ""
        self._bus = None
        self._cargarMontaje(montajePath or self.MONTAJE_JSON)
        if placement:
            self.placement = placement
        try:
            from smbus2 import SMBus
        except ImportError:
            self.reason = "smbus2 no instalado (pip install smbus2)"
            return
        try:
            self._bus = SMBus(bus)
            chip = self._bus.read_byte_data(address, self.REG_CHIP_ID)
            if chip != self.CHIP_ID:
                self.reason = (f"chip id 0x{chip:02X} != 0xA0 en "
                               f"0x{address:02X} (¿dirección correcta?)")
                return
            self._bus.write_byte_data(address, self.REG_OPR_MODE,
                                      self.MODE_CONFIG)
            time.sleep(0.025)
            self._bus.write_byte_data(address, self.REG_PWR_MODE, 0x00)
            self._bus.write_byte_data(address, self.REG_UNIT_SEL, 0x00)
            if self.placement in self.PLACEMENTS:
                cfg, sgn = self.PLACEMENTS[self.placement]
                self._bus.write_byte_data(address,
                                          self.REG_AXIS_MAP_CONFIG, cfg)
                self._bus.write_byte_data(address,
                                          self.REG_AXIS_MAP_SIGN, sgn)
            self._bus.write_byte_data(address, self.REG_OPR_MODE,
                                      self.MODE_NDOF)
            time.sleep(0.03)
            self.available =True
        except OSError as exc:
            self.reason = f"I2C sin respuesta ({exc})"

    def _readS16(self, reg, n):
        raw = self._bus.read_i2c_block_data(self.address, reg, 2 * n)
        out = []
        for i in range(n):
            v = raw[2 * i] | (raw[2 * i + 1] << 8)
            out.append(v - 65536 if v > 32767 else v)
        return out

    def _cargarMontaje(self, ruta):
        self.placement = None
        self.montajeDesde = "default"
        try:
            import json
            with open(ruta, encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            return
        self.placement = m.get("placement")
        self.HEADING_SIGN = float(m.get("headingSigno", self.HEADING_SIGN))
        self.PITCH_SIGN = float(m.get("pitchSigno", self.PITCH_SIGN))
        self.PITCH_DESDE = m.get("pitchDesde", self.PITCH_DESDE)
        self.PITCH_OFFSET_DEG = float(m.get("pitchOffsetDeg", 0.0))
        self.montajeDesde = ruta

    def readEulerCrudo(self):
        try:
            h, r, p = self._readS16(self.REG_EULER_H, 3)
        except OSError:
            return None
        return h / 16.0, r / 16.0, p / 16.0

    def readEuler(self):
        """(heading, roll, pitch) en rad ya en la convencion del
        sistema: heading + = giro a la izquierda, pitch + = cabeza hacia
        abajo (camara y ultrasonicos viendo al piso)."""
        crudo = self.readEulerCrudo()
        if crudo is None:
            return None
        h, r, p = crudo
        cabeceo = p if self.PITCH_DESDE == "pitch" else r
        otro = r if self.PITCH_DESDE == "pitch" else p
        pitchDeg = self.PITCH_SIGN * cabeceo - self.PITCH_OFFSET_DEG
        return (math.radians(self.HEADING_SIGN * h), math.radians(otro),
                math.radians(pitchDeg))

    def readLinearAccel(self):
        try:
            ax, ay, az = self._readS16(self.REG_LIN_ACCEL, 3)
        except OSError:
            return None
        return (ax / 100.0, ay / 100.0, az / 100.0)

    def calibration(self):
        try:
            s = self._bus.read_byte_data(self.address,
                                         self.REG_CALIB_STAT)
        except OSError:
            return None
        return {"sys": (s >> 6) & 3, "gyro": (s >> 4) & 3,
                "accel": (s >> 2) & 3, "mag": s & 3}


class StepDetector:
    def __init__(self, threshold=1.8, resetLevel=0.8,
                 minIntervalS=0.30):
        self.threshold = threshold
        self.resetLevel =resetLevel
        self.minIntervalS = minIntervalS
        self._armado = True
        self._ultimoPaso = -1e9
        self.stepsTotal = 0

    # pico de accel = paso
    def update(self, accelMag, t):
        papu = False
        if (self._armado and accelMag > self.threshold
                and (t - self._ultimoPaso) >= self.minIntervalS):
            papu = True
            self._armado = False
            self._ultimoPaso= t
            self.stepsTotal += 1
        elif not self._armado and accelMag < self.resetLevel:
            self._armado = True
        return papu


class MotionEstimator:
    def __init__(self, imu=None, yawEstimator=None, stepLengthM=0.68,
                 stepDetector=None):
        self.imu = imu if (imu is not None and imu.available) else None
        self.yawEstimator = yawEstimator
        self.stepLengthM= stepLengthM
        self.steps = stepDetector or StepDetector()
        self._prevHeading = None
        self._t = 0.0

    @property
    def source(self):
        if self.imu is not None:
            return "imu"
        return "flujo" if self.yawEstimator is not None else "ninguno"

    # imu primero, si no flujo optico
    def update(self, frameBgr, dt):
        self._t += dt
        info = {"source": "ninguno", "pitch_deg": None,
                "steps_total": self.steps.stepsTotal, "calib": None}

        if self.imu is not None:
            euler = self.imu.readEuler()
            accel = (self.imu.readLinearAccel()
                     if euler is not None else None)
            if euler is not None:
                heading, _roll, pitch = euler
                if self._prevHeading is None:
                    deltaRot = 0.0
                else:
                    deltaRot =_wrapAng(heading - self._prevHeading)
                self._prevHeading = heading
                deltaTrans = 0.0
                if accel is not None:
                    mag = math.sqrt(accel[0] ** 2 + accel[1] ** 2
                                    + accel[2] ** 2)
                    if self.steps.update(mag, self._t):
                        deltaTrans = self.stepLengthM
                info.update(source="imu",
                            pitch_deg=math.degrees(pitch),
                            steps_total=self.steps.stepsTotal,
                            calib=self.imu.calibration())
                return (MotionEstimate(delta_trans=deltaTrans,
                                       delta_rot=deltaRot, dt=dt), info)

        if self.yawEstimator is not None and frameBgr is not None:
            deltaRot, ok = self.yawEstimator.update(frameBgr)
            info["source"] = "flujo" if ok else "ninguno"
            return (MotionEstimate(delta_trans=0.0, delta_rot=deltaRot,
                                   dt=dt), info)

        return MotionEstimate(0.0, 0.0, dt), info


# arma el estimador, con flujo de respaldo
def makeMotionEstimator(useImu=True, imuBus=1, imuAddr=0x28,
                        stepLengthM=0.68, hfovDeg=70.0,
                        say=print):
    from vision_bridge import VisualYawEstimator
    imu = None
    if useImu:
        imu = Bno055(bus=imuBus, address=imuAddr)
        if imu.available:
            say("Odometría: IMU BNO055 activa (giro fusionado + pasos).")
            cal = imu.calibration()
            if cal and cal["mag"] < 2:
                say("IMU: mueva el dispositivo en forma de 8 para "
                    "calibrar la brújula.")
        else:
            say(f"Odometría: IMU no disponible ({imu.reason}); "
                "se usa giro por flujo óptico y avance 0.")
    return MotionEstimator(imu=imu,
                           yawEstimator=VisualYawEstimator(
                               hfovDeg=hfovDeg),
                           stepLengthM=stepLengthM)


def _promedio(imu, segundos=1.5):
    vals = []
    t0 = time.time()
    while time.time() - t0 < segundos:
        c = imu.readEulerCrudo()
        if c is not None:
            vals.append(c)
        time.sleep(0.05)
    n = max(1, len(vals))
    return tuple(sum(v[i] for v in vals) / n for i in range(3))


def _difAng(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def calibrarMontaje(imu, ruta=Bno055.MONTAJE_JSON, entrada=input):
    """guia de 3 pasos con los lentes puestos: mide que campo del
    bno055 es el cabeceo, sus signos y el cero. guarda el json que
    Bno055 carga solo al arrancar."""
    import json
    entrada("1) cabeza DERECHA viendo al horizonte, quieto. Enter...")
    base = _promedio(imu)
    entrada("2) inclina la cabeza hacia ABAJO (~30 grados) y Enter...")
    abajo = _promedio(imu)
    entrada("3) regresa al frente y GIRA la cabeza a la IZQUIERDA "
            "(~45 grados). Enter...")
    izq = _promedio(imu)

    dRoll, dPitch = _difAng(abajo[1], base[1]), _difAng(abajo[2], base[2])
    desde = "pitch" if abs(dPitch) >= abs(dRoll) else "roll"
    delta = dPitch if desde == "pitch" else dRoll
    if abs(delta) < 8:
        raise RuntimeError("casi no cambio al agachar la cabeza: "
                           "revisa que la imu este bien fija")
    signo = 1.0 if delta > 0 else -1.0
    offset = signo * (base[2] if desde == "pitch" else base[1])
    dHead = _difAng(izq[0], base[0])
    headingSigno = 1.0 if dHead > 0 else -1.0
    m = {"placement": imu.placement, "pitchDesde": desde,
         "pitchSigno": signo, "pitchOffsetDeg": round(offset, 2),
         "headingSigno": headingSigno}
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
    print(f"listo: {m} -> {ruta}")
    return m


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="utilidades de la imu bno055")
    p.add_argument("--calibrar", action="store_true",
                   help="calibracion guiada del montaje en los lentes")
    p.add_argument("--placement", default=None,
                   choices=sorted(Bno055.PLACEMENTS))
    p.add_argument("--addr", type=lambda s: int(s, 0), default=0x28)
    a = p.parse_args()
    imu = Bno055(address=a.addr, placement=a.placement)
    if not imu.available:
        raise SystemExit(f"imu no disponible: {imu.reason}")
    if a.calibrar:
        calibrarMontaje(imu)
    else:
        while True:
            e = imu.readEuler()
            if e:
                print("heading %+6.1f  pitch %+5.1f (+ = abajo)"
                      % (math.degrees(e[0]), math.degrees(e[2])), end="\r")
            time.sleep(0.1)
