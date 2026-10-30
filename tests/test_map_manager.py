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
from navigation_feed import (buildNavigationFrame, directionDistances,
                             sectorRisks, safetyAlerts, needsReplan,
                             effectiveFrontDistance)
from vision_bridge import (fuseObservations, detectionRisk,
                           groundDistanceFromRow)


# pinta ecos varias veces pa pasar el umbral
def pintarPared(mgrOGrid, pose, rangeM, bearings, passes=3):
    for _ in range(passes):
        obs = [Observation(range_m=rangeM, bearing=b, sigma_r=0.05)
               for b in bearings]
        mgrOGrid.updateFromScan(pose, obs)


class TestChunkCoordinates(unittest.TestCase):

    def setUp(self):
        self.mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0,
                                     resolution=0.05)

    def testCellAndChunkOfOrigin(self):
        gx, gy = self.mgr.worldToCell(0.01, 0.01)
        self.assertEqual((gx, gy), (0, 0))
        index, local = self.mgr.cellToChunk(gx, gy)
        self.assertEqual(index, (0, 0))
        self.assertEqual(local, (0, 0))

    def testNegativeWorldCoordinates(self):
        gx, gy = self.mgr.worldToCell(-0.01, -0.01)
        self.assertEqual((gx, gy), (-1, -1))
        index, local = self.mgr.cellToChunk(gx, gy)
        self.assertEqual(index, (-1, -1))
        self.assertEqual(local, (self.mgr.cells - 1, self.mgr.cells - 1))

    def testChunkBoundary(self):
        self.assertEqual(self.mgr.chunkOfPose(3.99, 0.0), (0, 0))
        self.assertEqual(self.mgr.chunkOfPose(4.01, 0.0), (1, 0))


class TestRayAcrossChunks(unittest.TestCase):

    def testRayWritesCellsInTwoChunks(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        pintarPared(mgr, (0.5, 0.5, 0.0), 3.0, [0.0])
        self.assertIn((0, 0), mgr.chunks)
        self.assertIn((1, 0), mgr.chunks)
        gx, gy = mgr.worldToCell(3.5, 0.5)
        lo, _ = mgr._cellValues(gx, gy)
        self.assertGreater(lo, 1.0)
        gx, gy = mgr.worldToCell(2.0, 0.5)
        lo, _ = mgr._cellValues(gx, gy)
        self.assertLess(lo, 0.0)

    def testCastDistanceAcrossBoundary(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        pintarPared(mgr, (0.5, 0.5, 0.0), 3.0, [-0.1, 0.0, 0.1])
        d = mgr.castDistance(0.5, 0.5, 0.0, maxRange=4.0)
        self.assertAlmostEqual(d, 3.0, delta=0.15)


class TestChunkEvents(unittest.TestCase):

    def testNewVsKnownChunk(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        evs = mgr.updatePosition((0.5, 0.5, 0.0))
        self.assertEqual(evs[0]["type"], "chunk_change")
        self.assertFalse(evs[0]["known"])
        evs = mgr.updatePosition((2.5, 0.5, 0.0))
        self.assertFalse(evs[0]["known"])
        evs= mgr.updatePosition((0.5, 0.5, 0.0))
        self.assertTrue(evs[0]["known"])

    def testScanCreatedChunkStillCountsAsNew(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        pintarPared(mgr, (0.5, 0.5, 0.0), 3.0, [0.0])
        evs = mgr.updatePosition((2.5, 0.5, 0.0))
        self.assertFalse(evs[0]["known"])

    def testNoEventWithinSameChunk(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        evs = mgr.updatePosition((1.5, 1.5, 0.0))
        self.assertEqual(evs, [])

    def testDoorCrossingCreatesRoom(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        landmarks = [{"x": 1.0, "y": 0.5, "label": "door",
                      "confirmed": True}]
        evs = mgr.updatePosition((0.9, 0.5, 0.0), landmarks)
        self.assertTrue(any(e["type"] == "door_crossed" for e in evs))
        self.assertIsNotNone(mgr.currentRoom)

    def testRoomRecognizedOnReentry(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0,
                                doorCooldownSteps=0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        mgr.setRoomLabel("cocina")
        mgr.updatePosition((2.5, 0.5, 0.0))
        mgr.currentRoom = None
        evs = mgr.updatePosition((0.5, 0.5, 0.0))
        self.assertTrue(any(e["type"] == "room_recognized"
                            and e["room"] == "cocina" for e in evs))


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="chunks_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # guardar y cargar no pierde nada
    def testRoundtripLogOddsHazardRoom(self):
        mgr = ChunkedMapManager(mapDir=self.dir, chunkSizeM=2.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        mgr.setRoomLabel("pasillo")
        pintarPared(mgr, (0.5, 0.5, 0.0), 1.5, [0.0])
        for _ in range(HAZARD_MIN_HITS):
            mgr.updateFromScan((0.5, 0.5, 0.0),
                               [Observation(range_m=1.0, bearing=0.5,
                                            is_hazard=True)])
        mgr.saveAll()

        mgr2 = ChunkedMapManager(mapDir=self.dir, chunkSizeM=2.0)
        self.assertIn((0, 0), mgr2.knownChunkIndices())
        gx, gy = mgr2.worldToCell(2.0, 0.5)
        lo, _ = mgr2._cellValues(gx, gy)
        self.assertGreater(lo, 1.0)
        ex = 0.5 + 1.0 * math.cos(0.5)
        ey =0.5 + 1.0 * math.sin(0.5)
        gx, gy = mgr2.worldToCell(ex, ey)
        _, hz = mgr2._cellValues(gx, gy)
        self.assertGreaterEqual(hz, HAZARD_MIN_HITS)
        evs = mgr2.updatePosition((0.5, 0.5, 0.0))
        self.assertTrue(evs[0]["known"])
        self.assertTrue(any(e.get("room") == "pasillo" for e in evs))

    def testIncompatibleGeometryRejected(self):
        mgr = ChunkedMapManager(mapDir=self.dir, chunkSizeM=2.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        mgr.saveAll()
        with self.assertRaises(ValueError):
            ChunkedMapManager(mapDir=self.dir, chunkSizeM=4.0)

    def testEvictionSavesDirtyChunks(self):
        mgr = ChunkedMapManager(mapDir=self.dir, chunkSizeM=2.0,
                                keepLoaded=4)
        for k in range(12):
            x = 0.5 + 2.0 * k
            mgr.updatePosition((x, 0.5, 0.0))
            pintarPared(mgr, (x, 0.5, 0.0), 1.0, [0.0], passes=1)
        self.assertLessEqual(len(mgr.chunks), 4)
        self.assertGreater(len(mgr.diskIndex), 0)
        gx, gy = mgr.worldToCell(1.5, 0.5)
        lo, _ = mgr._cellValues(gx, gy)
        self.assertGreater(lo, 0.0)

    def testLandmarkRoundtripWithEkf(self):
        slam = EKFSlam(0, 0, 0)
        for _ in range(3):
            slam.correct([Observation(range_m=2.0, bearing=0.0,
                                      label="door")])
        mgr = ChunkedMapManager(mapDir=self.dir)
        mgr.saveLandmarks(slam.exportLandmarks())

        slam2 = EKFSlam(0, 0, 0)
        slam2.importLandmarks(mgr.loadLandmarks())
        puertas = [lm for lm in slam2.getLandmarks()
                   if lm["label"] == "door"]
        self.assertEqual(len(puertas), 1)
        self.assertTrue(puertas[0]["confirmed"])
        slam2.predict(__import__("interfaces").MotionEstimate(0.1, 0.0))
        trAntes = float(np.trace(slam2.P))
        slam2.correct([Observation(range_m=1.9, bearing=0.0,
                                   label="door")])
        self.assertLess(float(np.trace(slam2.P)), trAntes)


class TestCostmapAndHazards(unittest.TestCase):

    def testCostmapAcrossChunksAndUnknownFill(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        pintarPared(mgr, (0.5, 0.5, 0.0), 3.0, [-0.1, 0.0, 0.1])
        cm = mgr.localCostmap((3.0, 0.5, 0.0), sizeM=3.0)
        self.assertEqual(cm.shape[0], 2)
        self.assertGreater((cm[0] > 0.7).sum(), 0)
        self.assertGreater((cm[0] < 0.3).sum(), 0)
        self.assertGreater((cm[0] == 0.5).sum(), 0)

    def testHazardNeedsMinHits(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        hzObs = [Observation(range_m=1.0, bearing=0.0, is_hazard=True)]
        mgr.updateFromScan((0.5, 0.5, 0.0), hzObs)
        gx, gy = mgr.worldToCell(1.5, 0.5)
        self.assertFalse(mgr.isBlockedCell(gx, gy))
        mgr.updateFromScan((0.5, 0.5, 0.0), hzObs)
        self.assertTrue(mgr.isBlockedCell(gx, gy))

    def testHazardBlocksCastDistance(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        for _ in range(HAZARD_MIN_HITS):
            mgr.updateFromScan((0.5, 0.5, 0.0),
                               [Observation(range_m=1.0, bearing=0.0,
                                            is_hazard=True)])
        d = mgr.castDistance(0.5, 0.5, 0.0, maxRange=4.0)
        self.assertLess(d, 1.2)

    def testOccupancyGridHasSameInterface(self):
        grid = OccupancyGrid(sizeM=6.0, resolution=0.05,
                             origin=(1.0, 1.0))
        pintarPared(grid, (0.5, 0.5, 0.0), 2.0, [-0.1, 0.0, 0.1])
        d = grid.castDistance(0.5, 0.5, 0.0, maxRange=4.0)
        self.assertAlmostEqual(d, 2.0, delta=0.15)
        cm = grid.localCostmap((0.5, 0.5, 0.0), sizeM=5.0)
        self.assertEqual(cm.shape[0], 2)
        self.assertGreater((cm[0] > 0.7).sum(), 0)

    def testGridRayClippedAtBorderKeepsFreeCells(self):
        grid = OccupancyGrid(sizeM=2.0, resolution=0.05,
                             origin=(1.0, 1.0))
        grid.updateFromScan((0.0, 0.0, 0.0),
                            [Observation(range_m=3.5, bearing=0.0)],
                            maxRange=4.0)
        gx, gy = grid.worldToGrid(0.5, 0.0)
        self.assertLess(grid.logOdds[gy, gx], 0.0)


class TestNavigationFeed(unittest.TestCase):

    def _mgrConPared(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        pintarPared(mgr, (0.5, 0.5, 0.0), 1.2,
                    [b * 0.1 for b in range(-4, 5)])
        return mgr

    def testDirectionDistances(self):
        mgr = self._mgrConPared()
        dFront, dLeft, dRight = directionDistances(
            mgr, (0.5, 0.5, 0.0), maxRange=4.0)
        self.assertLess(dFront, 1.5)
        self.assertGreater(dLeft, 2.0)
        self.assertGreater(dRight, 2.0)

    def testSectorRisks(self):
        dets = [
            VisionDetection(bearing=0.0, label="chair", range_est=1.5),
            VisionDetection(bearing=0.9, label="person", range_est=2.0,
                            moving=True),
            VisionDetection(bearing=-0.9, label="hoyo", range_est=1.0,
                            hazard=True),
        ]
        rf, rl, rr = sectorRisks(dets)
        self.assertEqual(rf, 2)
        self.assertEqual(rl, 3)
        self.assertEqual(rr, 3)

    def testFarDetectionsIgnoredForRisk(self):
        dets = [VisionDetection(bearing=0.0, label="chair",
                                range_est=5.0)]
        self.assertEqual(sectorRisks(dets), (0, 0, 0))

    def testNavigationFrameAndAlerts(self):
        mgr = self._mgrConPared()
        slam = EKFSlam(0.5, 0.5, 0.0)
        dets = [VisionDetection(bearing=0.0, label="table",
                                range_est=0.6)]
        frame = buildNavigationFrame(slam, mgr, dets,
                                     chunk=(0, 0), room="sala")
        self.assertEqual(frame.costmap.shape[0], 2)
        self.assertLess(frame.d_front, 1.5)
        self.assertEqual(frame.risk_front, 2)
        self.assertEqual(frame.chunk, (0, 0))
        alertas = safetyAlerts(frame)
        self.assertTrue(any(p == 0 for p, _ in alertas))

    def testNoAlertsInOpenSpace(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        slam = EKFSlam(0.5, 0.5, 0.0)
        frame = buildNavigationFrame(slam, mgr, [])
        self.assertEqual(safetyAlerts(frame), [])

    # pared al lado = precaucion, al frente = critico
    def testLateralWallIsCautionNotCritical(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        pintarPared(mgr, (2.0, 2.0, 0.0), 0.6,
                    [math.radians(90) + b * 0.1 for b in range(-5, 6)])
        slam = EKFSlam(2.0, 2.0, 0.0)
        frame = buildNavigationFrame(slam, mgr, [])
        self.assertLess(frame.d_left, 0.8)
        alertas = safetyAlerts(frame)
        self.assertTrue(alertas)
        self.assertTrue(all(p == 1 for p, _ in alertas))
        self.assertFalse(needsReplan(alertas))
        slamFrente = EKFSlam(2.0, 2.0, math.radians(90))
        frameF = buildNavigationFrame(slamFrente, mgr, [])
        papuAlerts = safetyAlerts(frameF)
        self.assertTrue(any(p == 0 and "al frente" in m for p, m in papuAlerts))
        self.assertTrue(needsReplan(papuAlerts))

    def testLateralHighRiskDetectionIsCritical(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        slam = EKFSlam(0.5, 0.5, 0.0)
        dets = [VisionDetection(bearing=math.radians(60), label="person",
                                range_est=0.7, moving=True)]
        alertas = safetyAlerts(buildNavigationFrame(slam, mgr, dets))
        self.assertTrue(any(p == 0 and "person" in m and "izquierda" in m
                            for p, m in alertas))

    def testAlertLabelOnlyWhenDetectionIsNearest(self):
        mgr = self._mgrConPared()
        slam = EKFSlam(0.5, 0.5, 0.0)
        dets = [VisionDetection(bearing=0.0, label="potted plant",
                                range_est=3.5)]
        alertas = safetyAlerts(buildNavigationFrame(slam, mgr, dets))
        self.assertTrue(alertas)
        self.assertTrue(all("obstaculo" in m for _, m in alertas))
        frame = buildNavigationFrame(slam, mgr, dets)
        self.assertAlmostEqual(effectiveFrontDistance(frame),
                               frame.d_front)

    def testAlertTextIsStableToRangeNoise(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        slam = EKFSlam(0.5, 0.5, 0.0)
        a = safetyAlerts(buildNavigationFrame(
            slam, mgr, [VisionDetection(0.0, "chair", range_est=0.57)]))
        b = safetyAlerts(buildNavigationFrame(
            slam, mgr, [VisionDetection(0.0, "chair", range_est=0.63)]))
        self.assertEqual(a, b)
        self.assertIn("60 centimetros", a[0][1])


class TestGroundGeometryAndHazardFusion(unittest.TestCase):

    def testGroundDistanceMonotonic(self):
        vfov = math.radians(55)
        dCerca = groundDistanceFromRow(470, 480, vfov, 1.55, 10.0)
        dLejos = groundDistanceFromRow(300, 480, vfov, 1.55, 10.0)
        self.assertIsNotNone(dCerca)
        self.assertIsNotNone(dLejos)
        self.assertLess(dCerca, dLejos)

    def testAboveHorizonIsNone(self):
        vfov = math.radians(55)
        self.assertIsNone(groundDistanceFromRow(0, 480, vfov, 1.55, 0.0))

    def testGroundDistanceFlatCameraGeometry(self):
        vfov = math.radians(60)
        f = 0.5 * 480 / math.tan(vfov / 2)
        row = 400
        esperado = 1.5 / math.tan(math.atan((row - 240) / f))
        got = groundDistanceFromRow(row, 480, vfov, 1.5, 0.0)
        self.assertAlmostEqual(got, esperado, places=6)

    def testDetectionRiskTable(self):
        self.assertEqual(detectionRisk(VisionDetection(
            bearing=0, label="person", moving=True)), 3)
        self.assertEqual(detectionRisk(VisionDetection(
            bearing=0, label="hoyo", hazard=True)), 3)
        self.assertEqual(detectionRisk(VisionDetection(
            bearing=0, label="escalera")), 3)
        self.assertEqual(detectionRisk(VisionDetection(
            bearing=0, label="chair")), 2)
        self.assertEqual(detectionRisk(VisionDetection(
            bearing=0, label="wall")), 1)

    def testFuseHazardBecomesHazardObservation(self):
        dets = [VisionDetection(bearing=0.2, label="coladera",
                                range_est=1.8, hazard=True)]
        obs = fuseObservations(dets, [])
        self.assertEqual(len(obs), 1)
        self.assertTrue(obs[0].is_hazard)
        self.assertEqual(obs[0].label, "coladera")
        self.assertEqual(obs[0].min_confirm_hits, 4)

    def testHazardNeverPairsWithUltrasonic(self):
        from interfaces import UltrasonicReading
        dets = [VisionDetection(bearing=0.0, label="hoyo",
                                range_est=1.8, hazard=True)]
        us = [UltrasonicReading(range_m=1.8, sensor_bearing=0.0)]
        obs = fuseObservations(dets, us)
        hz = [o for o in obs if o.is_hazard]
        self.assertEqual(len(hz), 1)
        self.assertGreater(hz[0].sigma_r, 0.1)
        self.assertTrue(any(o.label == "eco_us" for o in obs))


if __name__ == "__main__":
    unittest.main()


class TestRelocalizacion(unittest.TestCase):
    """al abrir un mapa guardado, los chunks de disco no se anuncian
    como conocidos hasta que el slam confirma re-localizacion."""

    def _mapaGuardado(self, tmpdir):
        mgr = ChunkedMapManager(mapDir=tmpdir, chunkSizeM=2.0)
        mgr.updatePosition((0.5, 0.5, 0.0))
        mgr.updatePosition((2.5, 0.5, 0.0))
        mgr.setRoomLabel("pasillo")
        mgr.saveAll()

    def test_pendiente_no_presume_conocido(self):
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(prefix="reloc_")
        try:
            self._mapaGuardado(tmpdir)
            mgr2 = ChunkedMapManager(mapDir=tmpdir, chunkSizeM=2.0)
            mgr2.marcarRelocPendiente()
            evs = mgr2.updatePosition((0.5, 0.5, 0.0))
            cambio = [e for e in evs if e["type"] == "chunk_change"][0]
            self.assertFalse(cambio["known"])          # honesto: no sabemos
            self.assertFalse(any(e["type"] == "room_recognized"
                                 for e in evs))
            # confirmando (slam falso con correcciones) ya si es conocido
            class SlamFalso:
                correccionesTotales = 5
            self.assertTrue(mgr2.checarReloc(SlamFalso()))
            self.assertFalse(mgr2.checarReloc(SlamFalso()))  # solo una vez
            evs = mgr2.updatePosition((2.5, 0.5, 0.0))
            cambio = [e for e in evs if e["type"] == "chunk_change"][0]
            self.assertTrue(cambio["known"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sin_mapa_guardado_no_cambia_nada(self):
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=2.0)
        evs = mgr.updatePosition((0.5, 0.5, 0.0))
        self.assertFalse(evs[0]["known"])
        class SlamFalso:
            correccionesTotales = 99
        self.assertFalse(mgr.checarReloc(SlamFalso()))   # nada pendiente
