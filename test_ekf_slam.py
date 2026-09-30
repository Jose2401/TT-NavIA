"""
test_ekf_slam.py
Pruebas unitarias / funcionales del módulo SLAM (EKF), de la fusión de
observaciones (vision_bridge) y de la simulación end-to-end.

Ejecutar con:
    python -m unittest test_ekf_slam -v
"""
import math
import unittest
import numpy as np

from ekf_slam import EKFSlam, wrap_angle
from interfaces import (MotionEstimate, Observation, UltrasonicReading,
                        VisionDetection)
from vision_bridge import fuse_observations, free_space_rays


class TestEKFSlam(unittest.TestCase):

    def test_initial_pose(self):
        slam = EKFSlam(1.0, 2.0, math.radians(30))
        out = slam.get_output()
        self.assertAlmostEqual(out.x, 1.0)
        self.assertAlmostEqual(out.y, 2.0)

    def test_predict_moves_forward(self):
        slam = EKFSlam(0, 0, 0)
        slam.predict(MotionEstimate(delta_trans=1.0, delta_rot=0.0))
        self.assertAlmostEqual(slam.mu[0], 1.0, places=3)
        self.assertAlmostEqual(slam.mu[1], 0.0, places=3)

    def test_predict_rotation(self):
        slam = EKFSlam(0, 0, 0)
        slam.predict(MotionEstimate(delta_trans=0.0, delta_rot=math.radians(90)))
        self.assertAlmostEqual(slam.mu[2], math.radians(90), places=3)

    def test_predict_increases_uncertainty(self):
        slam = EKFSlam(0, 0, 0)
        tr0 = np.trace(slam.P)
        slam.predict(MotionEstimate(delta_trans=0.5, delta_rot=0.1))
        tr1 = np.trace(slam.P)
        self.assertGreater(tr1, tr0)

    def test_new_landmark_registered(self):
        slam = EKFSlam(0, 0, 0)
        slam.correct([Observation(range_m=2.0, bearing=0.0)])
        self.assertEqual(len(slam.landmarks), 1)
        # Con una sola observación aún es candidato, no confirmado
        self.assertFalse(slam.landmarks[0].confirmed)

    def test_candidate_confirms_after_repeated_sightings(self):
        slam = EKFSlam(0, 0, 0)
        slam.correct([Observation(range_m=2.0, bearing=0.0)])
        slam.correct([Observation(range_m=2.0, bearing=0.0)])
        self.assertTrue(slam.landmarks[0].confirmed)

    def test_spurious_detection_does_not_correct(self):
        """Una detección aislada (falso positivo) no debe mover la pose."""
        slam = EKFSlam(0, 0, 0)
        mu_before = slam.mu.copy()
        slam.correct([Observation(range_m=3.0, bearing=1.0)])
        np.testing.assert_allclose(slam.mu, mu_before)

    def test_spurious_candidate_is_pruned(self):
        slam = EKFSlam(0, 0, 0, candidate_ttl=5)
        slam.correct([Observation(range_m=3.0, bearing=1.0)])
        self.assertEqual(len(slam.landmarks), 1)
        for _ in range(10):
            slam.step(MotionEstimate(0.0, 0.0), [])
        self.assertEqual(len(slam.landmarks), 0)

    def test_correction_reduces_uncertainty(self):
        slam = EKFSlam(0, 0, 0)
        slam.correct([Observation(range_m=2.0, bearing=0.0)])   # candidato ~(2,0)
        slam.correct([Observation(range_m=2.0, bearing=0.0)])   # confirma
        slam.predict(MotionEstimate(delta_trans=0.1, delta_rot=0.0))
        tr_before = np.trace(slam.P)
        # Observación del landmark ya confirmado -> corrige
        slam.correct([Observation(range_m=1.9, bearing=0.0)])
        tr_after = np.trace(slam.P)
        self.assertLess(tr_after, tr_before)

    def test_covariance_stays_symmetric_psd(self):
        slam = EKFSlam(0, 0, 0)
        for i in range(50):
            slam.step(MotionEstimate(delta_trans=0.05, delta_rot=0.01),
                      [Observation(range_m=2.0 - 0.05 * i, bearing=0.0)])
        np.testing.assert_allclose(slam.P, slam.P.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(slam.P)
        self.assertTrue(np.all(eigvals > -1e-12))

    def test_per_observation_noise_is_respected(self):
        """Un rango impreciso (monocular) debe corregir MENOS que uno
        preciso (ultrasonido) ante la misma innovación."""
        def displacement(sigma_r):
            slam = EKFSlam(0, 0, 0, confirm_hits=1)
            slam.correct([Observation(range_m=2.0, bearing=0.0)])
            slam.predict(MotionEstimate(delta_trans=0.1, delta_rot=0.0))
            mu_before = slam.mu.copy()
            slam.correct([Observation(range_m=2.05, bearing=0.0,
                                      sigma_r=sigma_r)])
            return abs(slam.mu[0] - mu_before[0])

        self.assertGreater(displacement(0.05), displacement(0.60))

    def test_dead_reckoning_without_observations(self):
        slam = EKFSlam(0, 0, 0)
        slam.predict(MotionEstimate(delta_trans=0.5, delta_rot=0.0))
        slam.correct([])   # sin observaciones -> solo predicción
        self.assertEqual(slam.cycles_without_correction, 1)
        self.assertLess(slam.confidence, 1.0)

    def test_confidence_drops_progressively(self):
        slam = EKFSlam(0, 0, 0)
        confs = []
        for _ in range(5):
            slam.step(MotionEstimate(delta_trans=0.1, delta_rot=0.0), [])
            confs.append(slam.confidence)
        self.assertTrue(all(confs[i] > confs[i + 1] for i in range(4)))

    def test_invalid_observation_ignored(self):
        slam = EKFSlam(0, 0, 0)
        slam.correct([Observation(range_m=-1.0, bearing=0.0),
                      Observation(range_m=float("nan"), bearing=0.0)])
        self.assertEqual(len(slam.landmarks), 0)

    def test_output_structure(self):
        slam = EKFSlam(0, 0, 0)
        out = slam.get_output()
        d = out.as_dict()
        for key in ("pose", "covarianza", "timestamp", "confianza"):
            self.assertIn(key, d)
        self.assertEqual(len(d["covarianza"]), 3)
        self.assertEqual(len(d["covarianza"][0]), 3)

    def test_full_cycle_step(self):
        slam = EKFSlam(0, 0, 0)
        out = slam.step(MotionEstimate(delta_trans=0.2, delta_rot=0.0),
                        [Observation(range_m=2.0, bearing=0.0)])
        self.assertAlmostEqual(out.x, 0.2, places=2)

    def test_theta_wraps(self):
        slam = EKFSlam(0, 0, math.radians(170))
        slam.predict(MotionEstimate(delta_trans=0.0,
                                    delta_rot=math.radians(30)))
        self.assertLessEqual(slam.mu[2], math.pi)
        self.assertAlmostEqual(slam.mu[2], wrap_angle(math.radians(200)),
                               places=6)


class TestNoDuplicateLandmarks(unittest.TestCase):
    """El caso reportado en la prueba con cámara real: el rango monocular
    fluctúa ~30% y el mismo objeto NO debe aparecer duplicado en el mapa."""

    def test_noisy_monocular_range_yields_single_landmark(self):
        import random
        random.seed(7)
        slam = EKFSlam(0, 0, 0)
        for _ in range(30):
            r = 2.0 * (1.0 + random.gauss(0.0, 0.3))
            slam.step(MotionEstimate(0.0, 0.0),
                      [Observation(range_m=max(0.3, r), bearing=0.0,
                                   label="chair", sigma_r=0.6,
                                   min_confirm_hits=4)])
        chairs = [lm for lm in slam.landmarks if lm.label == "chair"]
        self.assertEqual(len(chairs), 1)
        self.assertTrue(chairs[0].confirmed)

    def test_different_labels_stay_separate(self):
        slam = EKFSlam(0, 0, 0)
        for _ in range(5):
            slam.correct([
                Observation(range_m=2.0, bearing=0.0, label="chair"),
                Observation(range_m=2.4, bearing=0.0, label="table",
                            sigma_r=0.1),
            ])
        labels = sorted(lm.label for lm in slam.landmarks)
        self.assertEqual(labels, ["chair", "table"])

    def test_flickering_false_positive_never_confirms(self):
        """Un falso positivo de YOLO que dura 2 frames no debe volverse
        un obstáculo 'inventado' (min_confirm_hits=4 monocular)."""
        slam = EKFSlam(0, 0, 0, candidate_ttl=10)
        fake = Observation(range_m=3.0, bearing=0.5, label="vase",
                           sigma_r=0.9, min_confirm_hits=4)
        slam.correct([fake])
        slam.correct([fake])
        self.assertFalse(any(lm.confirmed for lm in slam.landmarks))
        for _ in range(15):
            slam.step(MotionEstimate(0.0, 0.0), [])
        self.assertEqual(len(slam.landmarks), 0)   # podado

    def test_dedup_merges_drifted_duplicates(self):
        from ekf_slam import Landmark
        slam = EKFSlam(0, 0, 0)
        a = Landmark(2.0, 0.0, label="chair"); a.confirmed = True
        b = Landmark(2.4, 0.3, label="chair"); b.confirmed = True
        b.seen_count = 3
        slam.landmarks = [a, b]
        slam.correct([])
        chairs = [lm for lm in slam.landmarks if lm.label == "chair"]
        self.assertEqual(len(chairs), 1)
        self.assertEqual(chairs[0].seen_count, 4)

    def test_observations_for_map_snap(self):
        """Las observaciones asociadas se anclan a la posición consolidada
        del landmark: la silla produce UNA marca en el mapa, en la
        posición del landmark, aunque el rango crudo fluctúe."""
        import random
        random.seed(3)
        slam = EKFSlam(0, 0, 0)
        for _ in range(10):
            r = 2.0 * (1.0 + random.gauss(0.0, 0.25))
            slam.correct([Observation(range_m=max(0.3, r), bearing=0.0,
                                      label="chair", sigma_r=0.5)])
        snapped = slam.observations_for_map()
        self.assertEqual(len(snapped), 1)          # una silla = una marca
        self.assertEqual(snapped[0].label, "chair")
        # La marca coincide con el landmark consolidado, no con el rango
        # crudo de esta iteración
        lm = [l for l in slam.landmarks if l.confirmed][0]
        expected_r = math.hypot(lm.x - slam.mu[0], lm.y - slam.mu[1])
        self.assertAlmostEqual(snapped[0].range_m, expected_r, places=6)
        self.assertEqual(len([l for l in slam.landmarks
                              if l.label == "chair"]), 1)

    def test_generic_echo_upgraded_by_vision_label(self):
        """Si el eco y la visión coinciden en el mismo punto, el landmark
        hereda la etiqueta del objeto (útil para rutas: 'door')."""
        slam = EKFSlam(0, 0, 0)
        slam.correct([Observation(range_m=2.0, bearing=0.0, label="eco_us",
                                  min_confirm_hits=3)])
        slam.correct([Observation(range_m=2.0, bearing=0.0, label="door")])
        self.assertEqual(len(slam.landmarks), 1)
        self.assertEqual(slam.landmarks[0].label, "door")


class TestFusion(unittest.TestCase):

    def test_vision_plus_ultrasonic_uses_us_range(self):
        dets = [VisionDetection(bearing=0.05, label="chair", range_est=3.0)]
        us = [UltrasonicReading(range_m=1.8, sensor_bearing=0.0)]
        obs = fuse_observations(dets, us)
        self.assertEqual(len(obs), 1)
        self.assertAlmostEqual(obs[0].range_m, 1.8)
        self.assertAlmostEqual(obs[0].bearing, 0.05)   # bearing de visión
        self.assertLess(obs[0].sigma_r, 0.1)

    def test_vision_without_us_uses_monocular_with_high_sigma(self):
        dets = [VisionDetection(bearing=1.0, label="chair", range_est=3.0)]
        us = [UltrasonicReading(range_m=1.8, sensor_bearing=0.0)]
        obs = fuse_observations(dets, us)
        vis_obs = [o for o in obs if o.label == "chair"]
        self.assertEqual(len(vis_obs), 1)
        self.assertAlmostEqual(vis_obs[0].range_m, 3.0)
        self.assertGreater(vis_obs[0].sigma_r, 0.5)

    def test_unmatched_us_becomes_wide_bearing_observation(self):
        obs = fuse_observations([], [UltrasonicReading(range_m=2.0,
                                                       sensor_bearing=0.7)])
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].label, "eco_us")
        self.assertGreater(obs[0].sigma_phi, math.radians(5))

    def test_moving_objects_excluded(self):
        dets = [VisionDetection(bearing=0.0, label="person",
                                range_est=2.0, moving=True)]
        self.assertEqual(fuse_observations(dets, []), [])

    def test_invalid_us_ignored_for_ekf_but_gives_free_ray(self):
        us = [UltrasonicReading(range_m=4.0, sensor_bearing=0.0,
                                max_range=4.0, valid=False)]
        self.assertEqual(fuse_observations([], us), [])
        rays = free_space_rays(us)
        self.assertEqual(len(rays), 1)
        self.assertGreater(rays[0].range_m, 4.0)


class TestVisualYaw(unittest.TestCase):
    """El estimador de giro por flujo óptico (modo solo-cámara) debe medir
    el paneo con el signo y la magnitud correctos."""

    def _textured(self):
        import cv2
        rng = np.random.default_rng(0)
        big = (rng.random((600, 1400)) * 255).astype(np.uint8)
        big = cv2.GaussianBlur(big, (5, 5), 0)
        return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    def test_pan_left_is_positive(self):
        from vision_bridge import VisualYawEstimator
        big = self._textured()
        est = VisualYawEstimator(hfov_deg=70)
        est.update(big[:, 400:1040])
        # la escena se desplaza +20 px (a la derecha) = giro a la izquierda
        dyaw, ok = est.update(big[:, 380:1020])
        self.assertTrue(ok)
        f = (640 / 2) / math.tan(math.radians(35))
        self.assertAlmostEqual(dyaw, math.atan(20 / f), places=3)

    def test_pan_right_is_negative(self):
        from vision_bridge import VisualYawEstimator
        big = self._textured()
        est = VisualYawEstimator(hfov_deg=70)
        est.update(big[:, 380:1020])
        dyaw, ok = est.update(big[:, 400:1040])
        self.assertTrue(ok)
        self.assertLess(dyaw, 0.0)

    def test_no_texture_reports_not_ok(self):
        from vision_bridge import VisualYawEstimator
        flat = np.full((480, 640, 3), 127, np.uint8)
        est = VisualYawEstimator(hfov_deg=70)
        est.update(flat)
        dyaw, ok = est.update(flat)
        self.assertFalse(ok)
        self.assertEqual(dyaw, 0.0)


class TestSimulationEndToEnd(unittest.TestCase):

    def test_simulation_accuracy(self):
        """El recorrido completo por el cuarto simulado debe mantener un
        error de trayectoria y orientación pequeños, y confianza sana.

        Nota sobre el umbral: al corregir el bug de geometría de
        sim_world.ray_distance (los rayos rebotaban en la EXTENSIÓN
        infinita de los muros), el banco de simulación se volvió más
        exigente: ahora hay ecos densos y reales y los rangos
        monoculares son la única corrección del EKF (los ecos ya no
        corrigen, ver Observation.can_correct). Referencia re-medida
        sobre 6 semillas: ATE prom ~0.27 m, máx ~0.44 m, orientación
        ~3 grados."""
        from simulation import run_simulation
        res = run_simulation(steps=250, seed=1, verbose=False,
                             croquis_path="croquis_test.png")
        self.assertLess(res["ate"], 0.35)
        self.assertLess(math.degrees(res["orientation_error"]), 10.0)
        self.assertGreater(res["slam"].confidence, 0.5)
        self.assertGreaterEqual(len(res["slam"].confirmed_landmarks()), 3)

    def test_simulation_survives_occlusion(self):
        """Con pérdida del 60% de las detecciones visuales el sistema debe
        seguir funcionando (dead-reckoning + ultrasonido)."""
        from simulation import run_simulation
        res = run_simulation(steps=250, seed=2, drop_prob=0.6, verbose=False,
                             croquis_path="croquis_test.png")
        self.assertLess(res["ate"], 0.45)


if __name__ == "__main__":
    unittest.main()
