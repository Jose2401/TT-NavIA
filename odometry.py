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

    HEADING_SIGN = -1.0
    PITCH_SIGN = 1.0

    def __init__(self, bus=1, address=0x28):
        self.address = address
        self.available= False
        self.reason = ""
        self._bus = None
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

    def readEuler(self):
        try:
            h, r, p = self._readS16(self.REG_EULER_H, 3)
        except OSError:
            return None
        aRad = math.radians(1.0 / 16.0)
        return (self.HEADING_SIGN * h * aRad, r * aRad,
                self.PITCH_SIGN * p * aRad)

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
