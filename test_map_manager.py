"""
test_map_manager.py
Pruebas del mapa global por chunks (map_manager), de la capa de peligros
de piso, del costmap para navegación (navigation_feed) y de la
persistencia/re-localización.

Ejecutar con:
    python -m unittest test_map_manager -v
"""
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from ekf_slam import EKFSlam
from interfaces import Observation, VisionDetection
from map_manager import ChunkedMapManager, MapChunk
from occupancy_grid import OccupancyGrid, HAZARD_MIN_HITS
from navigation_feed import (build_navigation_frame, direction_distances,
                             sector_risks, safety_alerts)
from vision_bridge import (fuse_observations, detection_risk,
                           ground_distance_from_row)


def paint_wall(mgr_or_grid, pose, range_m, bearings, passes=3):
    """Pinta ecos precisos repetidos para superar el umbral log-odds."""
    for _ in range(passes):
        obs = [Observation(range_m=range_m, bearing=b, sigma_r=0.05)
               for b in bearings]
        mgr_or_grid.update_from_scan(pose, obs)


class TestChunkCoordinates(unittest.TestCase):

    def setUp(self):
        self.mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0,
                                     resolution=0.05)

    def test_cell_and_chunk_of_origin(self):
        gx, gy = self.mgr.world_to_cell(0.01, 0.01)
        self.assertEqual((gx, gy), (0, 0))
        index, local = self.mgr.cell_to_chunk(gx, gy)
        self.assertEqual(index, (0, 0))
        self.assertEqual(local, (0, 0))

    def test_negative_world_coordinates(self):
        """El plano se extiende a coordenadas negativas (exteriores):
        floor division, no truncamiento hacia cero."""
        gx, gy = self.mgr.world_to_cell(-0.01, -0.01)
        self.assertEqual((gx, gy), (-1, -1))
        index, local = self.mgr.cell_to_chunk(gx, gy)
        self.assertEqual(index, (-1, -1))
        self.assertEqual(local, (self.mgr.cells - 1, self.mgr.cells - 1))

    def test_chunk_boundary(self):
        self.assertEqual(self.mgr.chunk_of_pose(3.99, 0.0), (0, 0))
        self.assertEqual(self.mgr.chunk_of_pose(4.01, 0.0), (1, 0))


class TestRayAcrossChunks(unittest.TestCase):

    def test_ray_writes_cells_in_two_chunks(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        # usuario en el chunk (0,0), eco a 3 m: el rayo termina en (1,0)
        paint_wall(mgr, (0.5, 0.5, 0.0), 3.0, [0.0])
        self.assertIn((0, 0), mgr.chunks)
        self.assertIn((1, 0), mgr.chunks)
        # celda final ocupada en el chunk vecino
        gx, gy = mgr.world_to_cell(3.5, 0.5)
        lo, _ = mgr._cell_values(gx, gy)
        self.assertGreater(lo, 1.0)
        # celdas intermedias libres
        gx, gy = mgr.world_to_cell(2.0, 0.5)
        lo, _ = mgr._cell_values(gx, gy)
        self.assertLess(lo, 0.0)

    def test_cast_distance_across_boundary(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        paint_wall(mgr, (0.5, 0.5, 0.0), 3.0, [-0.1, 0.0, 0.1])
        d = mgr.cast_distance(0.5, 0.5, 0.0, max_range=4.0)
        self.assertAlmostEqual(d, 3.0, delta=0.15)


class TestChunkEvents(unittest.TestCase):

    def test_new_vs_known_chunk(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        evs = mgr.update_position((0.5, 0.5, 0.0))
        self.assertEqual(evs[0]["type"], "chunk_change")
        self.assertFalse(evs[0]["known"])          # primera vez: nuevo
        evs = mgr.update_position((2.5, 0.5, 0.0))
        self.assertFalse(evs[0]["known"])          # vecino: nuevo
        evs = mgr.update_position((0.5, 0.5, 0.0))
        self.assertTrue(evs[0]["known"])           # regreso: conocido

    def test_scan_created_chunk_still_counts_as_new(self):
        """El scan puede crear el chunk un instante antes de que el
        usuario entre; para el usuario sigue siendo zona nueva."""
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        mgr.update_position((0.5, 0.5, 0.0))
        paint_wall(mgr, (0.5, 0.5, 0.0), 3.0, [0.0])   # crea el chunk (1,0)
        evs = mgr.update_position((2.5, 0.5, 0.0))
        self.assertFalse(evs[0]["known"])

    def test_no_event_within_same_chunk(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        mgr.update_position((0.5, 0.5, 0.0))
        evs = mgr.update_position((1.5, 1.5, 0.0))
        self.assertEqual(evs, [])

    def test_door_crossing_creates_room(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        mgr.update_position((0.5, 0.5, 0.0))
        landmarks = [{"x": 1.0, "y": 0.5, "label": "door",
                      "confirmed": True}]
        evs = mgr.update_position((0.9, 0.5, 0.0), landmarks)
        self.assertTrue(any(e["type"] == "door_crossed" for e in evs))
        self.assertIsNotNone(mgr.current_room)

    def test_room_recognized_on_reentry(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0,
                                door_cooldown_steps=0)
        mgr.update_position((0.5, 0.5, 0.0))
        mgr.set_room_label("cocina")
        mgr.update_position((2.5, 0.5, 0.0))       # sale (hereda cuarto)
        mgr.current_room = None                     # olvida el contexto
        evs = mgr.update_position((0.5, 0.5, 0.0))  # regresa
        self.assertTrue(any(e["type"] == "room_recognized"
                            and e["room"] == "cocina" for e in evs))


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="chunks_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_log_odds_hazard_room(self):
        mgr = ChunkedMapManager(map_dir=self.dir, chunk_size_m=2.0)
        mgr.update_position((0.5, 0.5, 0.0))
        mgr.set_room_label("pasillo")
        paint_wall(mgr, (0.5, 0.5, 0.0), 1.5, [0.0])
        for _ in range(HAZARD_MIN_HITS):
            mgr.update_from_scan((0.5, 0.5, 0.0),
                                 [Observation(range_m=1.0, bearing=0.5,
                                              is_hazard=True)])
        mgr.save_all()

        mgr2 = ChunkedMapManager(map_dir=self.dir, chunk_size_m=2.0)
        self.assertIn((0, 0), mgr2.known_chunk_indices())
        # log-odds sobrevive (float16 en disco: tolerancia amplia)
        gx, gy = mgr2.world_to_cell(2.0, 0.5)
        lo, _ = mgr2._cell_values(gx, gy)
        self.assertGreater(lo, 1.0)
        # capa de peligros sobrevive
        ex = 0.5 + 1.0 * math.cos(0.5)
        ey = 0.5 + 1.0 * math.sin(0.5)
        gx, gy = mgr2.world_to_cell(ex, ey)
        _, hz = mgr2._cell_values(gx, gy)
        self.assertGreaterEqual(hz, HAZARD_MIN_HITS)
        # el cuarto se recuerda al volver a entrar
        evs = mgr2.update_position((0.5, 0.5, 0.0))
        self.assertTrue(evs[0]["known"])
        self.assertTrue(any(e.get("room") == "pasillo" for e in evs))

    def test_incompatible_geometry_rejected(self):
        mgr = ChunkedMapManager(map_dir=self.dir, chunk_size_m=2.0)
        mgr.update_position((0.5, 0.5, 0.0))
        mgr.save_all()
        with self.assertRaises(ValueError):
            ChunkedMapManager(map_dir=self.dir, chunk_size_m=4.0)

    def test_eviction_saves_dirty_chunks(self):
        mgr = ChunkedMapManager(map_dir=self.dir, chunk_size_m=2.0,
                                keep_loaded=4)
        # camina lejos: se crean muchos chunks, los lejanos se descargan
        for k in range(12):
            x = 0.5 + 2.0 * k
            mgr.update_position((x, 0.5, 0.0))
            paint_wall(mgr, (x, 0.5, 0.0), 1.0, [0.0], passes=1)
        self.assertLessEqual(len(mgr.chunks), 4)
        # lo descargado quedó en disco y sigue siendo consultable
        self.assertGreater(len(mgr.disk_index), 0)
        gx, gy = mgr.world_to_cell(1.5, 0.5)      # eco del primer chunk
        lo, _ = mgr._cell_values(gx, gy)
        self.assertGreater(lo, 0.0)

    def test_landmark_roundtrip_with_ekf(self):
        slam = EKFSlam(0, 0, 0)
        for _ in range(3):
            slam.correct([Observation(range_m=2.0, bearing=0.0,
                                      label="door")])
        mgr = ChunkedMapManager(map_dir=self.dir)
        mgr.save_landmarks(slam.export_landmarks())

        slam2 = EKFSlam(0, 0, 0)
        slam2.import_landmarks(mgr.load_landmarks())
        doors = [lm for lm in slam2.get_landmarks()
                 if lm["label"] == "door"]
        self.assertEqual(len(doors), 1)
        self.assertTrue(doors[0]["confirmed"])
        # el landmark restaurado corrige la pose desde el primer ciclo
        slam2.predict(__import__("interfaces").MotionEstimate(0.1, 0.0))
        tr_before = float(np.trace(slam2.P))
        slam2.correct([Observation(range_m=1.9, bearing=0.0,
                                   label="door")])
        self.assertLess(float(np.trace(slam2.P)), tr_before)


class TestCostmapAndHazards(unittest.TestCase):

    def test_costmap_across_chunks_and_unknown_fill(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        paint_wall(mgr, (0.5, 0.5, 0.0), 3.0, [-0.1, 0.0, 0.1])
        cm = mgr.local_costmap((3.0, 0.5, 0.0), size_m=3.0)
        self.assertEqual(cm.shape[0], 2)
        self.assertGreater((cm[0] > 0.7).sum(), 0)     # pared visible
        self.assertGreater((cm[0] < 0.3).sum(), 0)     # libre visible
        # zonas nunca vistas = 0.5 exacto
        self.assertGreater((cm[0] == 0.5).sum(), 0)

    def test_hazard_needs_min_hits(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        hz_obs = [Observation(range_m=1.0, bearing=0.0, is_hazard=True)]
        mgr.update_from_scan((0.5, 0.5, 0.0), hz_obs)
        gx, gy = mgr.world_to_cell(1.5, 0.5)
        self.assertFalse(mgr.is_blocked_cell(gx, gy))  # 1 hit: aún no
        mgr.update_from_scan((0.5, 0.5, 0.0), hz_obs)
        self.assertTrue(mgr.is_blocked_cell(gx, gy))   # 2 hits: peligro

    def test_hazard_blocks_cast_distance(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=2.0)
        for _ in range(HAZARD_MIN_HITS):
            mgr.update_from_scan((0.5, 0.5, 0.0),
                                 [Observation(range_m=1.0, bearing=0.0,
                                              is_hazard=True)])
        d = mgr.cast_distance(0.5, 0.5, 0.0, max_range=4.0)
        self.assertLess(d, 1.2)

    def test_occupancy_grid_has_same_interface(self):
        grid = OccupancyGrid(size_m=6.0, resolution=0.05,
                             origin=(1.0, 1.0))
        paint_wall(grid, (0.5, 0.5, 0.0), 2.0, [-0.1, 0.0, 0.1])
        d = grid.cast_distance(0.5, 0.5, 0.0, max_range=4.0)
        self.assertAlmostEqual(d, 2.0, delta=0.15)
        cm = grid.local_costmap((0.5, 0.5, 0.0), size_m=5.0)
        self.assertEqual(cm.shape[0], 2)
        self.assertGreater((cm[0] > 0.7).sum(), 0)

    def test_grid_ray_clipped_at_border_keeps_free_cells(self):
        """Antes, un rayo cuyo final caía fuera del mapa se descartaba
        completo; ahora las celdas internas sí se despejan."""
        grid = OccupancyGrid(size_m=2.0, resolution=0.05,
                             origin=(1.0, 1.0))
        grid.update_from_scan((0.0, 0.0, 0.0),
                              [Observation(range_m=3.5, bearing=0.0)],
                              max_range=4.0)
        gx, gy = grid.world_to_grid(0.5, 0.0)
        self.assertLess(grid.log_odds[gy, gx], 0.0)


class TestNavigationFeed(unittest.TestCase):

    def _mgr_with_wall(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        paint_wall(mgr, (0.5, 0.5, 0.0), 1.2,
                   [b * 0.1 for b in range(-4, 5)])
        return mgr

    def test_direction_distances(self):
        mgr = self._mgr_with_wall()
        d_front, d_left, d_right = direction_distances(
            mgr, (0.5, 0.5, 0.0), max_range=4.0)
        self.assertLess(d_front, 1.5)
        self.assertGreater(d_left, 2.0)
        self.assertGreater(d_right, 2.0)

    def test_sector_risks(self):
        dets = [
            VisionDetection(bearing=0.0, label="chair", range_est=1.5),
            VisionDetection(bearing=0.9, label="person", range_est=2.0,
                            moving=True),
            VisionDetection(bearing=-0.9, label="hoyo", range_est=1.0,
                            hazard=True),
        ]
        rf, rl, rr = sector_risks(dets)
        self.assertEqual(rf, 2)   # obstáculo estático = medio
        self.assertEqual(rl, 3)   # persona en movimiento = alto
        self.assertEqual(rr, 3)   # hoyo = alto

    def test_far_detections_ignored_for_risk(self):
        dets = [VisionDetection(bearing=0.0, label="chair",
                                range_est=5.0)]
        self.assertEqual(sector_risks(dets), (0, 0, 0))

    def test_navigation_frame_and_alerts(self):
        mgr = self._mgr_with_wall()
        slam = EKFSlam(0.5, 0.5, 0.0)
        dets = [VisionDetection(bearing=0.0, label="table",
                                range_est=0.6)]
        frame = build_navigation_frame(slam, mgr, dets,
                                       chunk=(0, 0), room="sala")
        self.assertEqual(frame.costmap.shape[0], 2)
        self.assertLess(frame.d_front, 1.5)
        self.assertEqual(frame.risk_front, 2)
        self.assertEqual(frame.chunk, (0, 0))
        alerts = safety_alerts(frame)
        self.assertTrue(any(p == 0 for p, _ in alerts))   # < 80 cm: crítico

    def test_no_alerts_in_open_space(self):
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        slam = EKFSlam(0.5, 0.5, 0.0)
        frame = build_navigation_frame(slam, mgr, [])
        self.assertEqual(safety_alerts(frame), [])


class TestGroundGeometryAndHazardFusion(unittest.TestCase):

    def test_ground_distance_monotonic(self):
        """Filas más abajo en la imagen = piso más cercano."""
        vfov = math.radians(55)
        d_near = ground_distance_from_row(470, 480, vfov, 1.55, 10.0)
        d_far = ground_distance_from_row(300, 480, vfov, 1.55, 10.0)
        self.assertIsNotNone(d_near)
        self.assertIsNotNone(d_far)
        self.assertLess(d_near, d_far)

    def test_above_horizon_is_none(self):
        vfov = math.radians(55)
        self.assertIsNone(ground_distance_from_row(0, 480, vfov, 1.55, 0.0))

    def test_ground_distance_flat_camera_geometry(self):
        """Con pitch 0 y fila conocida, la distancia debe coincidir con
        la geometría exacta d = h / tan(atan((row-c)/f))."""
        vfov = math.radians(60)
        f = 0.5 * 480 / math.tan(vfov / 2)
        row = 400
        expected = 1.5 / math.tan(math.atan((row - 240) / f))
        got = ground_distance_from_row(row, 480, vfov, 1.5, 0.0)
        self.assertAlmostEqual(got, expected, places=6)

    def test_detection_risk_table(self):
        self.assertEqual(detection_risk(VisionDetection(
            bearing=0, label="person", moving=True)), 3)
        self.assertEqual(detection_risk(VisionDetection(
            bearing=0, label="hoyo", hazard=True)), 3)
        self.assertEqual(detection_risk(VisionDetection(
            bearing=0, label="escalera")), 3)
        self.assertEqual(detection_risk(VisionDetection(
            bearing=0, label="chair")), 2)
        self.assertEqual(detection_risk(VisionDetection(
            bearing=0, label="wall")), 1)

    def test_fuse_hazard_becomes_hazard_observation(self):
        dets = [VisionDetection(bearing=0.2, label="coladera",
                                range_est=1.8, hazard=True)]
        obs = fuse_observations(dets, [])
        self.assertEqual(len(obs), 1)
        self.assertTrue(obs[0].is_hazard)
        self.assertEqual(obs[0].label, "coladera")
        self.assertEqual(obs[0].min_confirm_hits, 4)

    def test_hazard_never_pairs_with_ultrasonic(self):
        """El ultrasonido no ve hoyos: un eco en la misma dirección es
        OTRA cosa (pared del fondo), no el hoyo."""
        from interfaces import UltrasonicReading
        dets = [VisionDetection(bearing=0.0, label="hoyo",
                                range_est=1.8, hazard=True)]
        us = [UltrasonicReading(range_m=1.8, sensor_bearing=0.0)]
        obs = fuse_observations(dets, us)
        hz = [o for o in obs if o.is_hazard]
        self.assertEqual(len(hz), 1)
        self.assertGreater(hz[0].sigma_r, 0.1)   # sigue siendo monocular
        # el eco quedó libre y entra como eco_us aparte
        self.assertTrue(any(o.label == "eco_us" for o in obs))


if __name__ == "__main__":
    unittest.main()
