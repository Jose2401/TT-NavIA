"""
test_odometry.py
Pruebas del módulo de odometría (IMU BNO055 + dead reckoning peatonal)
SIN hardware: el detector de pasos y el estimador se prueban con trazas
sintéticas y una IMU falsa; el driver real solo se prueba en su ruta de
"no disponible" (en la Pi con el chip conectado se verifica en campo,
ver TODO de odometry.py).

Ejecutar con:
    python -m unittest test_odometry -v
"""
import math
import unittest

from odometry import Bno055, StepDetector, MotionEstimator


class TestStepDetector(unittest.TestCase):

    def _walk_trace(self, steps, period=0.5, dt=0.02):
        """Magnitud de aceleración sintética: un pico por paso."""
        out = []
        t = 0.0
        total = steps * period
        while t < total:
            phase = (t % period) / period
            mag = 2.5 if 0.1 < phase < 0.2 else 0.2   # pico breve
            out.append((mag, t))
            t += dt
        return out

    def test_counts_steps(self):
        det = StepDetector()
        count = sum(det.update(m, t) for m, t in self._walk_trace(8))
        self.assertEqual(count, 8)
        self.assertEqual(det.stepsTotal, 8)

    def test_sustained_peak_counts_once(self):
        """Un pico largo (sin regresar bajo reset_level) = UN paso."""
        det = StepDetector()
        count = sum(det.update(2.5, t / 100.0) for t in range(100))
        self.assertEqual(count, 1)

    def test_min_interval_rejects_bounces(self):
        """Dos picos a 0.1 s (rebote del sensor) = un solo paso."""
        det = StepDetector(minIntervalS=0.30)
        trace = [(2.5, 0.00), (0.2, 0.05), (2.5, 0.10), (0.2, 0.15),
                 (2.5, 0.50), (0.2, 0.55)]
        count = sum(det.update(m, t) for m, t in trace)
        self.assertEqual(count, 2)

    def test_quiet_signal_no_steps(self):
        det = StepDetector()
        count = sum(det.update(0.3, t / 50.0) for t in range(200))
        self.assertEqual(count, 0)


class FakeImu:
    """IMU falsa para probar MotionEstimator sin hardware."""

    def __init__(self, eulers, accels):
        self.available = True
        self.eulers = list(eulers)     # [(heading, roll, pitch), ...]
        self.accels = list(accels)
        self.i = 0

    def readEuler(self):
        v = self.eulers[min(self.i, len(self.eulers) - 1)]
        return v

    def readLinearAccel(self):
        v = self.accels[min(self.i, len(self.accels) - 1)]
        self.i += 1
        return v

    def calibration(self):
        return {"sys": 3, "gyro": 3, "accel": 3, "mag": 3}


class TestMotionEstimator(unittest.TestCase):

    def test_imu_heading_delta(self):
        imu = FakeImu(eulers=[(0.0, 0, 0.2), (math.radians(10), 0, 0.2)],
                      accels=[(0, 0, 0), (0, 0, 0)])
        est = MotionEstimator(imu=imu)
        m0, info0 = est.update(None, 0.1)
        self.assertEqual(info0["source"], "imu")
        self.assertAlmostEqual(m0.delta_rot, 0.0)       # primer ciclo
        m1, info1 = est.update(None, 0.1)
        self.assertAlmostEqual(m1.delta_rot, math.radians(10), places=6)
        self.assertAlmostEqual(info1["pitch_deg"],
                               math.degrees(0.2), places=4)

    def test_heading_wraps_at_pi(self):
        """De 179° a -179° es un giro de +2°, no de -358°."""
        imu = FakeImu(eulers=[(math.radians(179), 0, 0),
                              (math.radians(-179), 0, 0)],
                      accels=[(0, 0, 0)] * 2)
        est = MotionEstimator(imu=imu)
        est.update(None, 0.1)
        m, _ = est.update(None, 0.1)
        self.assertAlmostEqual(m.delta_rot, math.radians(2), places=6)

    def test_step_advances_step_length(self):
        imu = FakeImu(eulers=[(0, 0, 0)] * 4,
                      accels=[(0, 0, 0.2), (0, 0, 2.5),
                              (0, 0, 0.2), (0, 0, 0.2)])
        est = MotionEstimator(imu=imu, stepLengthM=0.7)
        dists = [est.update(None, 0.2)[0].delta_trans for _ in range(4)]
        self.assertEqual(dists.count(0.7), 1)           # un paso
        self.assertAlmostEqual(sum(dists), 0.7)

    def test_unavailable_imu_falls_back(self):
        class DeadImu:
            available = False
            reason = "sin chip"
        est = MotionEstimator(imu=DeadImu())
        self.assertIsNone(est.imu)
        m, info = est.update(None, 0.1)                 # sin frame: nada
        self.assertEqual(info["source"], "ninguno")
        self.assertEqual(m.delta_trans, 0.0)

    def test_io_error_mid_session_falls_back_that_cycle(self):
        class FlakyImu(FakeImu):
            def readEuler(self):
                return None                             # I2C falló
        est = MotionEstimator(imu=FlakyImu([], []))
        m, info = est.update(None, 0.1)
        self.assertEqual(info["source"], "ninguno")     # sin frame tampoco
        self.assertEqual(m.delta_rot, 0.0)


class TestBno055Driver(unittest.TestCase):

    def test_graceful_without_smbus2_or_chip(self):
        """En la laptop (sin smbus2/chip) el driver debe reportar
        available=False con una razón útil, nunca lanzar."""
        imu = Bno055()
        self.assertFalse(imu.available)
        self.assertTrue(imu.reason)


if __name__ == "__main__":
    unittest.main()
