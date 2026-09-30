"""
test_navigation.py
Pruebas del módulo de navegación: entorno de entrenamiento, formatos de
entrada (comandos) y salida (indicaciones), submódulo de dirección con
las frases exactas del documento, backends del modelo (SB3 vs numpy) y
lazo cerrado con el modelo entrenado.

Ejecutar con:
    python -m unittest test_navigation -v
"""
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


def _pick_model():
    for base in (FINAL_MODEL, CMP_MODEL):
        if os.path.exists(base + "_policy.npz"):
            return base
    return None


class TestNavEnv(unittest.TestCase):

    def test_spaces_and_reset(self):
        for mode, dim in (("doc", 8), ("extended", 24)):
            env = NavEnv(obs_mode=mode, seed=1)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (dim,))
            self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 1.0))

    def test_forward_into_wall_terminates_with_collision(self):
        env = NavEnv(obs_mode="doc", seed=2, sensor_noise=0)
        env.reset()
        # colocar al agente pegado a la pared oeste mirando hacia ella
        env.pose = [0.35, env.world.bounds[3] / 2.0, math.pi]
        env.prev_d_goal = env._d_goal()
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(term)
        self.assertTrue(info.get("collision"))
        self.assertLess(r, -50)

    def test_reaching_goal_gives_success(self):
        env = NavEnv(obs_mode="doc", seed=3, sensor_noise=0)
        env.reset()
        # goal justo al frente, sin obstáculos en medio
        x, y, th = env.pose
        env.goal = (x + 0.5 * math.cos(th), y + 0.5 * math.sin(th))
        env.world.statics = np.zeros((0, 3))
        env.world.movers = np.zeros((0, 5))
        env.world.hazards = np.zeros((0, 3))
        env.prev_d_goal = env._d_goal()
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(info.get("success"))
        self.assertGreater(r, 50)

    def test_hazard_fall(self):
        env = NavEnv(obs_mode="extended", seed=4, sensor_noise=0)
        env.reset()
        x, y, th = env.pose
        hx = x + 0.3 * math.cos(th)
        hy = y + 0.3 * math.sin(th)
        env.world.hazards = np.array([[hx, hy, 0.25]])
        obs, r, term, trunc, info = env.step(0)
        self.assertTrue(info.get("fell"))

    def test_world_generators_produce_valid_worlds(self):
        rng = np.random.default_rng(0)
        for gen in (gen_room, gen_two_rooms, gen_outdoor):
            world, start, goal = gen(rng)
            self.assertGreater(world.clearance_at(start[0], start[1]), 0.2)
            d = world.ray_distances(start[0], start[1],
                                    np.linspace(-math.pi, math.pi, 8))
            self.assertTrue(np.all(d > 0))


class TestCommandFormat(unittest.TestCase):
    """Formato de ENTRADA: las intenciones del documento (Alg. 9)."""

    def test_destination_phrases(self):
        for texto, destino in (("ve a la silla", "silla"),
                               ("llevame a la salida", "salida"),
                               ("navega hacia la puerta", "puerta"),
                               ("destino cocina", "cocina")):
            cmd = parse_command(texto)
            self.assertEqual(cmd.tipo, "navegar", texto)
            self.assertEqual(cmd.destino, destino, texto)

    def test_point_destination(self):
        cmd = parse_command("punto 3.5 -2")
        self.assertEqual(cmd.tipo, "navegar")
        self.assertEqual(cmd.punto, (3.5, -2.0))

    def test_session_intents(self):
        self.assertEqual(parse_command("pausa").tipo, "pausar")
        self.assertEqual(parse_command("continúa por favor").tipo, "reanudar")
        self.assertEqual(parse_command("detente").tipo, "detener")
        self.assertEqual(parse_command("dame otra ruta").tipo,
                         "ruta_alternativa")
        self.assertEqual(parse_command("¿dónde estoy?").tipo, "estado")
        self.assertEqual(parse_command("xyzzy").tipo, "desconocido")


class TestDirectionText(unittest.TestCase):
    """Submódulo de dirección (Alg. 5): frases y umbrales EXACTOS."""

    def test_forward_with_distance(self):
        self.assertEqual(direction_text(0, 7.4, 0.0), "Avance 4 metros")

    def test_slight_turn_at_45_or_less(self):
        self.assertEqual(direction_text(1, 8.0, math.radians(30)),
                         "Gire ligeramente a la izquierda")
        self.assertEqual(direction_text(2, 8.0, math.radians(-45)),
                         "Gire ligeramente a la derecha")

    def test_full_turn_over_45(self):
        self.assertEqual(direction_text(1, 8.0, math.radians(90)),
                         "Gire a la izquierda 90 grados")

    def test_stop(self):
        self.assertEqual(direction_text(3, 8.0, 0.0), "Deténgase")

    def test_near_destination_prefix(self):
        txt = direction_text(0, 3.0, 0.0)
        self.assertTrue(txt.startswith("Está cerca de su destino. "), txt)


@unittest.skipUnless(_pick_model(), "aún no hay modelo entrenado")
class TestPolicyBackends(unittest.TestCase):

    def test_numpy_matches_sb3(self):
        """El backend numpy (Raspberry) debe decidir IGUAL que SB3."""
        base = _pick_model()
        npz = NumpyPolicy(base + "_policy.npz")
        try:
            from navigation.navigator import SB3Policy
            sb3 = SB3Policy(base + ".zip")
        except Exception:
            self.skipTest("SB3 no disponible")
        rng = np.random.default_rng(0)
        dim = npz.layers[0][0].shape[1]
        agree = 0
        for _ in range(200):
            obs = rng.uniform(-1, 1, size=dim).astype(np.float32)
            if npz.predict(obs) == sb3.predict(obs):
                agree += 1
        self.assertGreaterEqual(agree, 198)   # tolerancia numérica mínima

    def test_model_beats_random_in_env(self):
        base = _pick_model()
        policy, obs_mode = load_policy(base)
        successes = 0
        for ep in range(20):
            env = NavEnv(obs_mode=obs_mode, seed=5000 + ep)
            obs, _ = env.reset()
            done = False
            while not done:
                obs, _r, term, trunc, info = env.step(policy.predict(obs))
                done = term or trunc
            successes += bool(info.get("success"))
        self.assertGreaterEqual(successes, 10)   # >=50% (aleatorio: ~0%)


@unittest.skipUnless(_pick_model(), "aún no hay modelo entrenado")
class TestNavigatorSession(unittest.TestCase):
    """Máquina de estados (Alg. 11) + formato de salida."""

    def _navigator(self):
        return Navigator(_pick_model(), repeat_s=0.0, decide_every_s=0.0)

    def _frame(self, x=0.0, y=0.0, theta=0.0, landmarks=None):
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from navigation_feed import build_navigation_frame
        slam = EKFSlam(x, y, theta)
        if landmarks:
            slam.import_landmarks(landmarks)
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        return build_navigation_frame(slam, mgr, []), mgr

    def test_full_session_flow(self):
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
        self.assertEqual(nav.update(frame, mgr), [])      # pausado: silencio
        self.assertEqual(nav.handle_command("reanuda")[0].text,
                         "Reanudando navegación.")
        self.assertEqual(nav.handle_command("detener")[0].text,
                         "Navegación detenida.")
        self.assertEqual(nav.status.session, "en_espera")

    def test_arrival_message(self):
        nav = self._navigator()
        frame, mgr = self._frame(x=2.8, y=0.0, landmarks=[
            {"x": 3.0, "y": 0.0, "label": "door", "seen_count": 5,
             "required_hits": 2}])
        nav.handle_command("ve a la puerta", frame)
        msgs = nav.update(frame, mgr)
        self.assertEqual(msgs[0].text, "Ha llegado a su destino.")
        self.assertEqual(nav.status.session, "en_espera")

    def test_unknown_destination(self):
        nav = self._navigator()
        frame, _ = self._frame()
        msgs = nav.handle_command("ve a la cocina", frame)
        self.assertEqual(nav.status.session, "en_espera")
        self.assertIn("conozco", msgs[0].text)

    def test_point_goal_and_observation_shapes(self):
        nav = self._navigator()
        frame, mgr = self._frame()
        nav.handle_command("punto 2 1", frame)
        self.assertEqual(nav.status.goal, (2.0, 1.0))
        for mode, dim in (("doc", 8), ("extended", 24)):
            obs, phi, d = build_observation(frame, mgr, (2.0, 1.0),
                                            mode, prev_action=3)
            self.assertEqual(obs.shape, (dim,))
            self.assertAlmostEqual(d, math.hypot(2, 1), places=5)


@unittest.skipUnless(_pick_model(), "aún no hay modelo entrenado")
class TestClosedLoopIntegration(unittest.TestCase):
    """El sistema COMPLETO: SLAM + mapa por chunks + navegador. El
    usuario simulado obedece las indicaciones y debe llegar al destino
    esquivando los muebles del cuarto."""

    def test_guided_walk_reaches_goal(self):
        import random
        random.seed(11)
        from sim_world import (SimulatedWorld, SimulatedUltrasonicArray,
                               simulated_vision)
        from sim_world import KeyboardUser
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from vision_bridge import fuse_observations, free_space_rays
        from navigation_feed import build_navigation_frame
        from navigation.nav_env import FORWARD_STEP, TURN_STEP

        world = SimulatedWorld.default_room()
        user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
        us = SimulatedUltrasonicArray(world)
        slam = EKFSlam(0.7, 0.7, 0.0)
        mgr = ChunkedMapManager(map_dir=None, chunk_size_m=4.0)
        nav = Navigator(_pick_model(), repeat_s=0.0, decide_every_s=0.0)

        goal = (5.0, 3.0)          # esquina opuesta, tras los muebles
        frame = None
        arrived = False
        for step in range(400):
            # obedecer la última acción decidida
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

            tp = tuple(user.true_pose)
            dets = simulated_vision(world, tp)
            readings = us.read(tp)
            out = slam.step(motion, fuse_observations(dets, readings))
            pose = (out.x, out.y, out.theta)
            mgr.update_from_scan(pose, slam.observations_for_map()
                                 + free_space_rays(readings))
            mgr.update_position(pose, slam.get_landmarks())
            frame = build_navigation_frame(slam, mgr, dets)

            if step == 10:
                nav.handle_command("punto 5 3", frame)
            msgs = nav.update(frame, mgr)
            if any(m.text == "Ha llegado a su destino." for m in msgs):
                arrived = True
                break

        self.assertTrue(arrived, "el usuario guiado no llegó al destino")
        err = math.hypot(user.true_pose[0] - goal[0],
                         user.true_pose[1] - goal[1])
        self.assertLess(err, 1.2)   # llegada real, no solo estimada


if __name__ == "__main__":
    unittest.main()
