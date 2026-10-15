"""
odometry.py — fuente del MotionEstimate (delta_d, delta_theta) pal ekf.
imu bno055 por i2c (fusion a bordo): el giro sale del heading fusionado
y el avance por CONTEO DE PASOS — integrar el acelerometro dos veces
deriva horrible en segundos, por eso contamos pasos (~0.68 m c/u).
sin imu cae solo a flujo optico (giro) y avance 0, igual que antes.
el bno tambien da el pitch real de la camara pal plano de piso.

TODO al conectar el hardware: direccion i2c 0x28 o 0x29 (ADR a 3.3V);
verificar HEADING_SIGN/PITCH_SIGN segun el montaje en los lentes
(girar a la IZQ => delta_rot POSITIVO, mirar al piso => pitch POSITIVO);
calibrar --step-length caminando 10 m; y mover el aparato en forma de 8
pa calibrar la brujula.
"""
import math
import time

from interfaces import MotionEstimate


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------------
# driver bno055 (i2c, smbus2), solo lo que usamos
# ----------------------------------------------------------------------
class Bno055:
    """lectura minima del bno055 en modo NDOF. si no hay smbus2 o chip,
    available=False y el sistema sigue con el fallback visual."""

    CHIP_ID = 0xA0
    REG_CHIP_ID = 0x00
    REG_OPR_MODE = 0x3D
    REG_PWR_MODE = 0x3E
    REG_SYS_TRIGGER = 0x3F
    REG_UNIT_SEL = 0x3B
    REG_EULER_H = 0x1A          # heading, roll, pitch: int16 LSB=1/16 grado
    REG_LIN_ACCEL = 0x28        # accel lineal sin gravedad, LSB=1/100 m/s2
    REG_CALIB_STAT = 0x35
    MODE_CONFIG = 0x00
    MODE_NDOF = 0x0C

    # OJO verificar el signo cuando montemos el chip en los lentes.
    # el heading del bno crece horario (brujula); nuestra theta al reves.
    HEADING_SIGN = -1.0
    PITCH_SIGN = 1.0

    def __init__(self, bus=1, address=0x28):
        self.address = address
        self.available = False
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
            # config -> unidades (grados, m/s2) -> NDOF
            self._bus.write_byte_data(address, self.REG_OPR_MODE,
                                      self.MODE_CONFIG)
            time.sleep(0.025)
            self._bus.write_byte_data(address, self.REG_PWR_MODE, 0x00)
            self._bus.write_byte_data(address, self.REG_UNIT_SEL, 0x00)
            self._bus.write_byte_data(address, self.REG_OPR_MODE,
                                      self.MODE_NDOF)
            time.sleep(0.03)
            self.available = True
        except OSError as exc:
            self.reason = f"I2C sin respuesta ({exc})"

    def _read_s16(self, reg, n):
        raw = self._bus.read_i2c_block_data(self.address, reg, 2 * n)
        out = []
        for i in range(n):
            v = raw[2 * i] | (raw[2 * i + 1] << 8)
            out.append(v - 65536 if v > 32767 else v)
        return out

    def read_euler(self):
        """(heading, roll, pitch) en rad ya con nuestros signos, o None
        si fallo la lectura (ese ciclo se cae al fallback)."""
        try:
            h, r, p = self._read_s16(self.REG_EULER_H, 3)
        except OSError:
            return None
        to_rad = math.radians(1.0 / 16.0)
        return (self.HEADING_SIGN * h * to_rad, r * to_rad,
                self.PITCH_SIGN * p * to_rad)

    def read_linear_accel(self):
        """(ax, ay, az) m/s2 sin gravedad, o None si falla."""
        try:
            ax, ay, az = self._read_s16(self.REG_LIN_ACCEL, 3)
        except OSError:
            return None
        return (ax / 100.0, ay / 100.0, az / 100.0)

    def calibration(self):
        """niveles 0-3; el magnetometro se calibra moviendo en 8."""
        try:
            s = self._bus.read_byte_data(self.address,
                                         self.REG_CALIB_STAT)
        except OSError:
            return None
        return {"sys": (s >> 6) & 3, "gyro": (s >> 4) & 3,
                "accel": (s >> 2) & 3, "mag": s & 3}


# ----------------------------------------------------------------------
# detector de pasos (puro software, probado sin hardware)
# ----------------------------------------------------------------------
class StepDetector:
    """un paso = pico de aceleracion sobre threshold, regreso bajo
    reset_level, y separacion minima (nadie da mas de ~3 pasos/s)."""

    def __init__(self, threshold=1.8, reset_level=0.8,
                 min_interval_s=0.30):
        self.threshold = threshold
        self.reset_level = reset_level
        self.min_interval_s = min_interval_s
        self._armed = True
        self._last_step_t = -1e9
        self.steps_total = 0

    def update(self, accel_mag, t):
        """-> True si este instante cuenta como paso."""
        step = False
        if (self._armed and accel_mag > self.threshold
                and (t - self._last_step_t) >= self.min_interval_s):
            step = True
            self._armed = False
            self._last_step_t = t
            self.steps_total += 1
        elif not self._armed and accel_mag < self.reset_level:
            self._armed = True
        return step


# ----------------------------------------------------------------------
# estimador de movimiento (imu + fallback visual)
# ----------------------------------------------------------------------
class MotionEstimator:
    """saca el MotionEstimate de cada ciclo.
    prioridad: imu (giro + pasos) > flujo optico (solo giro) > nada."""

    def __init__(self, imu=None, yaw_estimator=None, step_length_m=0.68,
                 step_detector=None):
        self.imu = imu if (imu is not None and imu.available) else None
        self.yaw_estimator = yaw_estimator
        self.step_length_m = step_length_m
        self.steps = step_detector or StepDetector()
        self._prev_heading = None
        self._t = 0.0

    @property
    def source(self):
        if self.imu is not None:
            return "imu"
        return "flujo" if self.yaw_estimator is not None else "ninguno"

    def update(self, frame_bgr, dt):
        self._t += dt
        info = {"source": "ninguno", "pitch_deg": None,
                "steps_total": self.steps.steps_total, "calib": None}

        if self.imu is not None:
            euler = self.imu.read_euler()
            accel = (self.imu.read_linear_accel()
                     if euler is not None else None)
            if euler is not None:
                heading, _roll, pitch = euler
                if self._prev_heading is None:
                    delta_rot = 0.0
                else:
                    delta_rot = _wrap(heading - self._prev_heading)
                self._prev_heading = heading
                delta_trans = 0.0
                if accel is not None:
                    mag = math.sqrt(accel[0] ** 2 + accel[1] ** 2
                                    + accel[2] ** 2)
                    if self.steps.update(mag, self._t):
                        delta_trans = self.step_length_m
                info.update(source="imu",
                            pitch_deg=math.degrees(pitch),
                            steps_total=self.steps.steps_total,
                            calib=self.imu.calibration())
                return (MotionEstimate(delta_trans=delta_trans,
                                       delta_rot=delta_rot, dt=dt), info)
            # lectura i2c fallida este ciclo: cae al fallback visual

        if self.yaw_estimator is not None and frame_bgr is not None:
            delta_rot, ok = self.yaw_estimator.update(frame_bgr)
            info["source"] = "flujo" if ok else "ninguno"
            return (MotionEstimate(delta_trans=0.0, delta_rot=delta_rot,
                                   dt=dt), info)

        return MotionEstimate(0.0, 0.0, dt), info


def make_motion_estimator(use_imu=True, imu_bus=1, imu_addr=0x28,
                          step_length_m=0.68, hfov_deg=70.0,
                          say=print):
    """fabrica de los mains: intenta la imu y SIEMPRE deja el flujo
    optico de respaldo."""
    from vision_bridge import VisualYawEstimator
    imu = None
    if use_imu:
        imu = Bno055(bus=imu_bus, address=imu_addr)
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
                           yaw_estimator=VisualYawEstimator(
                               hfov_deg=hfov_deg),
                           step_length_m=step_length_m)
