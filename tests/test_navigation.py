import math
import os
import unittest

import numpy as np

from navigation.nav_env import NavEnv, gen_room, gen_two_rooms, gen_outdoor
from navigation.navigator import (parse_command, direction_text,
                                  NumpyPolicy, load_policy, Navigator,
                                  build_observation)

CMP_MODEL = "models/nav_ppo_extended_cmp"
FINAL_MODEL = "models/nav_ppo"


def _elegirModelo():
    for base in (FINAL_MODEL, CMP_MODEL):
        if os.path.exists(base + "_policy.npz"):
            return base
    return None


class TestNavEnv(unittest.TestCase):

    def testSpacesAndReset(self):
        for mode, dim in (("doc", 8), ("extended", 24)):
            env = NavEnv(obs_mode=mode, seed=1)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (dim,))
            self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0))

    def testForwardIntoWallTerminatesWithCollision(self):
        env = NavEnv(obs_mode="doc", seed=2, sensor_noise=0)
        env.reset()
        env.pose = [0.35, env.world.bounds[3] / 2.0, math.pi]
        env.prev_d_goal = env._d_goal()
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(term)
        self.assertTrue(info.get("collision"))
        self.assertLess(r, -50)

    def testReachingGoalGivesSuccess(self):
        env = NavEnv(obs_mode="doc", seed=3, sensor_noise=0)
        env.reset()
        x, y, th = env.pose
        env.goal = (x + 0.5 * math.cos(th), y + 0.5 * math.sin(th))
        env.world.statics = np.zeros((0, 3))
        env.world.movers = np.zeros((0, 5))
        env.world.hazards= np.zeros((0, 3))
        env.prev_d_goal = env._d_goal()
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(info.get("success"))
        self.assertGreater(r, 50)

    def testHazardFall(self):
        env = NavEnv(obs_mode="extended", seed=4, sensor_noise=0)
        env.reset()
        x, y, th = env.pose
        hx = x + 0.3 * math.cos(th)
        hy =y + 0.3 * math.sin(th)
        env.world.hazards = np.array([[hx, hy, 0.25]])
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(info.get("fell"))

    def testWorldGeneratorsProduceValidWorlds(self):
        rng = np.random.default_rng(0)
        for gen in (gen_room, gen_two_rooms, gen_outdoor):
            world, start, goal = gen(rng)
            self.assertGreater(world.clearance_at(start[0], start[1]), 0.2)
            d = world.ray_distances(start[0], start[1],
                                    np.linspace(-math.pi, math.pi, 8))
            self.assertTrue(np.all(d > 0))


class TestCommandFormat(unittest.TestCase):

    def testDestinationPhrases(self):
        for texto, destino in (("ve a la silla", "silla"),
                               ("llevame a la salida", "salida"),
                               ("navega hacia la puerta", "puerta"),
                               ("destino cocina", "cocina")):
            cmd = parse_command(texto)
            self.assertEqual(cmd.tipo, "navegar", texto)
            self.assertEqual(cmd.destino, destino, texto)

    def testPointDestination(self):
        cmd = parse_command("punto 3.5 -2")
        self.assertEqual(cmd.tipo, "navegar")
        self.assertEqual(cmd.punto, (3.5, -2.0))

    def testSessionIntents(self):
        self.assertEqual(parse_command("pausa").tipo, "pausar")
        self.assertEqual(parse_command("continúa por favor").tipo, "reanudar")
        self.assertEqual(parse_command("detente").tipo, "detener")
        self.assertEqual(parse_command("dame otra ruta").tipo,
                         "ruta_alternativa")
        self.assertEqual(parse_command("¿dónde estoy?").tipo, "estado")
        self.assertEqual(parse_command("xyzzy").tipo, "desconocido")


class TestDirectionText(unittest.TestCase):

    def testForwardWithDistance(self):
        self.assertEqual(direction_text(0, 7.4, 0.0), "Avance 4 metros")

    def testSlightTurnAt45OrLess(self):
        self.assertEqual(direction_text(1, 8.0, math.radians(30)),
                         "Gire ligeramente a la izquierda")
        self.assertEqual(direction_text(2, 8.0, math.radians(-45)),
                         "Gire ligeramente a la derecha")

    def testFullTurnOver45(self):
        self.assertEqual(direction_text(1, 8.0, math.radians(90)),
                         "Gire a la izquierda 90 grados")

    def testStop(self):
        self.assertEqual(direction_text(3, 8.0, 0.0), "Deténgase")

    def testNearDestinationPrefix(self):
        txt = direction_text(0, 3.0, 0.0)
        self.assertTrue(txt.startswith("Está cerca de su destino. "), txt)


@unittest.skipUnless(_elegirModelo(), "aún no hay modelo entrenado")
class TestPolicyBackends(unittest.TestCase):

    def testNumpyMatchesSb3(self):
        base = _elegirModelo()
        npz = NumpyPolicy(base + "_policy.npz")
        try:
            from navigation.navigator import SB3Policy
            sb3 = SB3Policy(base + ".zip")
        except Exception:
            self.skipTest("SB3 no disponible")
        rng = np.random.default_rng(0)
        dim = npz.layers[0][0].shape[1]
        coinciden = 0
        for _ in range(200):
            obs = rng.uniform(-1, 1, size=dim).astype(np.float32)
            if npz.predict(obs) == sb3.predict(obs):
                coinciden += 1
        self.assertGreaterEqual(coinciden, 198)

    def testModelBeatsRandomInEnv(self):
        base = _elegirModelo()
        policy, obsMode = load_policy(base)
        exitos = 0
        for ep in range(20):
            env = NavEnv(obs_mode=obsMode, seed=5000 + ep)
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _r, term, trunc, info = env.step(policy.predict(obs))
                done = term or trunc
            exitos += bool(info.get("success"))
        self.assertGreaterEqual(exitos, 10)


@unittest.skipUnless(_elegirModelo(), "aún no hay modelo entrenado")
class TestNavigatorSession(unittest.TestCase):

    def _navigator(self):
        return Navigator(_elegirModelo(), repeat_s=0.0, decide_every_s=0.0)

    def _frame(self, x=0.0, y=0.0, theta=0.0, landmarks=None):
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from navigation_feed import buildNavigationFrame
        slam = EKFSlam(x, y, theta)
        if landmarks:
            slam.importLandmarks(landmarks)
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        return buildNavigationFrame(slam, mgr, []), mgr

    # flujo completo de sesion
    def testFullSessionFlow(self):
        nav = self._navigator()
        frame, mgr = self._frame(landmarks=[
            {"x": 3.0, "y": 0.0, "label": "door", "seen_count": 5,
             "required_hits": 2}])

        msgs = nav.handle_command("llevame a la salida", frame)
        self.assertEqual(msgs[0].text, "Ruta calculada. Iniciando navegación.")
        self.assertEqual(nav.status.session, "activo")
        self.assertEqual(nav.status.goal_label, "door")

        msgs = nav.update(frame, mgr)
        self.assertTrue(msgs)
        self.assertEqual(msgs[0].kind, "instruccion")
        self.assertEqual(msgs[0].priority, 1)
        self.assertIn(msgs[0].action, (0, 1, 2, 3))

        self.assertEqual(nav.handle_command("pausa")[0].text,
                         "Navegación pausada.")
        self.assertEqual(nav.update(frame, mgr), [])
        self.assertEqual(nav.handle_command("reanuda")[0].text,
                         "Reanudando navegación.")
        self.assertEqual(nav.handle_command("detener")[0].text,
                         "Navegación detenida.")
        self.assertEqual(nav.status.session, "en_espera")

    def testArrivalMessage(self):
        nav = self._navigator()
        frame, mgr = self._frame(x=2.8, y=0.0, landmarks=[
            {"x": 3.0, "y": 0.0, "label": "door", "seen_count": 5,
             "required_hits": 2}])
        nav.handle_command("ve a la puerta", frame)
        msgs = nav.update(frame, mgr)
        self.assertEqual(msgs[0].text, "Ha llegado a su destino.")
        self.assertEqual(nav.status.session, "en_espera")

    def testUnknownDestination(self):
        nav = self._navigator()
        frame, _ = self._frame()
        msgs = nav.handle_command("ve a la cocina", frame)
        self.assertEqual(nav.status.session, "en_espera")
        self.assertIn("conozco", msgs[0].text)

    def testPointGoalAndObservationShapes(self):
        nav = self._navigator()
        frame, mgr = self._frame()
        nav.handle_command("punto 2 1", frame)
        self.assertEqual(nav.status.goal, (2.0, 1.0))
        for mode, dim in (("doc", 8), ("extended", 24)):
            obs, phi, d = build_observation(frame, mgr, (2.0, 1.0),
                                            mode, prev_action=3)
            self.assertEqual(obs.shape, (dim,))
            self.assertAlmostEqual(d, math.hypot(2, 1), places=5)

    def _frameParedEnfrente(self, x=0.5, y=0.5, theta=0.0, dist=0.6):
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from navigation_feed import buildNavigationFrame
        from interfaces import Observation
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        for _ in range(3):
            mgr.updateFromScan((x, y, theta), [
                Observation(range_m=dist, bearing=theta + b * 0.1,
                            sigma_r=0.05) for b in range(-6, 7)])
        slam = EKFSlam(x, y, theta)
        return buildNavigationFrame(slam, mgr, []), mgr

    def testNeverSaysAdvanceWhenFrontIsCritical(self):
        from navigation_feed import safetyAlerts, needsReplan
        nav = self._navigator()
        frame, mgr = self._frameParedEnfrente(dist=0.6)
        nav.handle_command("punto 3 0.5", frame)
        alertas = safetyAlerts(frame)
        self.assertTrue(needsReplan(alertas))
        dirs = set()
        for _ in range(5):
            msgs = nav.update(frame, mgr, alertas)
            self.assertNotEqual(nav.status.last_action, 0)
            for m in msgs:
                self.assertNotIn("Avance", m.text)
            dirs.add(nav.status.last_action)
        self.assertEqual(len(dirs), 1)

    def testCriticalAlertForcesImmediateDecision(self):
        from navigation_feed import safetyAlerts
        nav = Navigator(_elegirModelo(), repeat_s=0.0, decide_every_s=1e6)
        frame, mgr = self._frameParedEnfrente(dist=0.6)
        nav.handle_command("punto 3 0.5", frame)
        self.assertTrue(nav.update(frame, mgr, []))
        self.assertEqual(nav.update(frame, mgr, []), [])
        msgs = nav.update(frame, mgr, safetyAlerts(frame))
        self.assertTrue(msgs)
        self.assertEqual(msgs[0].kind, "instruccion")

    def testForwardRepeatsAreCollapsed(self):
        nav = Navigator(_elegirModelo(), repeat_s=10.0, decide_every_s=0.0)
        frame, mgr = self._frame()
        nav.handle_command("punto 3 0", frame)
        dichos = []
        for _ in range(6):
            dichos += [m for m in nav.update(frame, mgr, [])
                       if m.action == 0]
        self.assertLessEqual(len(dichos), 1)


class TestVisionEgoMotion(unittest.TestCase):

    def testStaticObjectNotFlaggedMovingWhenTurning(self):
        try:
            from vision.motion import ObjectMotionTracker
        except ImportError:
            self.skipTest("vision/motion.py requiere OpenCV")
        from vision_bridge import shiftTrackerForYaw
        hfov = math.radians(70)
        w = 640
        yawPorFrame = math.radians(3)
        pxPorFrame = yawPorFrame * w / hfov

        def correr(compensar):
            tracker = ObjectMotionTracker()
            cx = 300.0
            flags = None
            for _ in range(5):
                if compensar:
                    shiftTrackerForYaw(tracker, yawPorFrame, hfov, w)
                det = [{"bbox": (int(cx - 30), 200, int(cx + 30), 320)}]
                flags = tracker.update(det)
                cx += pxPorFrame
            return flags[0]

        self.assertTrue(correr(compensar=False))
        self.assertFalse(correr(compensar=True))

    def testTrulyMovingObjectStillDetected(self):
        try:
            from vision.motion import ObjectMotionTracker
        except ImportError:
            self.skipTest("vision/motion.py requiere OpenCV")
        from vision_bridge import shiftTrackerForYaw
        hfov, w = math.radians(70), 640
        yaw = math.radians(3)
        tracker = ObjectMotionTracker()
        cx = 300.0
        for _ in range(5):
            shiftTrackerForYaw(tracker, yaw, hfov, w)
            flags = tracker.update([{"bbox": (int(cx - 30), 200,
                                              int(cx + 30), 320)}])
            cx += yaw * w / hfov + 12.0
        self.assertTrue(flags[0])


class TestRealMapWorlds(unittest.TestCase):

    # cuarto de 4x3 pintado con ecos buenos
    def _hacerMapa(self, tmpdir):
        import math as m
        from map_manager import ChunkedMapManager
        from interfaces import Observation
        mgr = ChunkedMapManager(mapDir=tmpdir, chunkSizeM=4.0)
        poses = [(x, y, th) for x in (1.0, 2.0, 3.0)
                 for y in (0.8, 1.5, 2.2)
                 for th in (0.0, m.pi / 2, m.pi, -m.pi / 2)]
        for pose in poses:
            x, y, th = pose
            obs = []
            for b in [k * 0.2 - 1.2 for k in range(13)]:
                ang = th + b
                mejor = 4.0
                for (wx0, wy0, wx1, wy1) in ((0, 0, 4, 0), (4, 0, 4, 3),
                                             (4, 3, 0, 3), (0, 3, 0, 0)):
                    ex, ey = wx1 - wx0, wy1 - wy0
                    den = m.cos(ang) * ey - m.sin(ang) * ex
                    if abs(den) < 1e-9:
                        continue
                    t = ((wx0 - x) * ey - (wy0 - y) * ex) / den
                    u= ((x - wx0) * m.sin(ang)
                        - (y - wy0) * m.cos(ang)) / -den
                    if t > 0 and 0 <= u <= 1:
                        mejor = min(mejor, t)
                obs.append(Observation(range_m=mejor, bearing=b,
                                       sigma_r=0.05))
            for _ in range(2):
                mgr.updateFromScan(pose, obs)
        mgr.saveAll()
        return mgr

    def testFactoryAndEpisode(self):
        import tempfile, shutil
        from navigation import real_maps
        tmpdir = tempfile.mkdtemp(prefix="realmap_test_")
        try:
            self._hacerMapa(tmpdir)
            real_maps._CACHE.clear()
            papu = real_maps.RealMapWorldFactory(tmpdir)
            self.assertGreater(len(papu.statics), 20)
            self.assertGreater(papu.free_area_m2(), 1.0)
            rng = np.random.default_rng(0)
            world, start, goal = papu.sample_episode(rng, movers=False)
            self.assertGreater(world.clearance_at(start[0], start[1]),
                               0.2)
            d = world.ray_distances(start[0], start[1],
                                    np.linspace(-math.pi, math.pi, 12),
                                    max_range=8.0)
            self.assertLess(d.min(), 4.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def testEnvUsesRealMap(self):
        import tempfile, shutil
        from navigation import real_maps
        tmpdir = tempfile.mkdtemp(prefix="realmap_env_")
        try:
            self._hacerMapa(tmpdir)
            real_maps._CACHE.clear()
            env = NavEnv(obs_mode="extended", seed=0,
                         real_map_dirs=[tmpdir], real_map_prob=1.0)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (24,))
            for _ in range(10):
                obs, _r, term, trunc, _i = env.step(
                    env.action_space.sample())
                if term or trunc:
                    break
            self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


@unittest.skipUnless(_elegirModelo(), "aún no hay modelo entrenado")
class TestClosedLoopIntegration(unittest.TestCase):

    def testGuidedWalkReachesGoal(self):
        import random
        random.seed(11)
        from sim_world import (SimulatedWorld, SimulatedUltrasonicArray,
                               simulatedVision)
        from sim_world import KeyboardUser
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from vision_bridge import fuseObservations, freeSpaceRays
        from navigation_feed import buildNavigationFrame
        from navigation.nav_env import FORWARD_STEP, TURN_STEP

        world = SimulatedWorld.defaultRoom()
        user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
        us = SimulatedUltrasonicArray(world)
        slam = EKFSlam(0.7, 0.7, 0.0)
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
        nav = Navigator(_elegirModelo(), repeat_s=0.0, decide_every_s=0.0)

        goal = (5.0, 3.0)
        frame = None
        llego = False
        for paso in range(400):
            forward = rot = 0.0
            act = nav.status.last_action
            if nav.status.session == "activo" and act is not None:
                if act == 0:
                    forward = FORWARD_STEP / 2
                elif act == 1:
                    rot = TURN_STEP / 2
                elif act == 2:
                    rot = -TURN_STEP / 2
            motion = user.command(forward, rot)

            tp = tuple(user.truePose)
            dets = simulatedVision(world, tp)
            lecturas = us.read(tp)
            out = slam.step(motion, fuseObservations(dets, lecturas))
            pose = (out.x, out.y, out.theta)
            mgr.updateFromScan(pose, slam.observationsForMap()
                               + freeSpaceRays(lecturas))
            mgr.updatePosition(pose, slam.getLandmarks())
            frame = buildNavigationFrame(slam, mgr, dets)

            if paso == 10:
                nav.handle_command("punto 5 3", frame)
            msgs = nav.update(frame, mgr)
            if any(m.text == "Ha llegado a su destino." for m in msgs):
                llego = True
                break

        self.assertTrue(llego, "el usuario guiado no llegó al destino")
        err = math.hypot(user.truePose[0] - goal[0],
                         user.truePose[1] - goal[1])
        self.assertLess(err, 1.2)


if __name__ == "__main__":
    unittest.main()
