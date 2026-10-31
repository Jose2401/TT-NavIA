"""
pruebas del modulo de alertas (alg. 7) y del planificador de ruta.
correr: python -m unittest tests.test_alertas_ruta
"""
import math
import os
import time
import unittest

from alertas import ModuloAlertas, BotonPullUp
from interfaces import VisionDetection, Observation
from map_manager import ChunkedMapManager
from navigation_feed import buildNavigationFrame
from ekf_slam import EKFSlam


def _frame(dets=None, mgr=None):
    slam = EKFSlam(0, 0, 0)
    mgr = mgr or ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
    return buildNavigationFrame(slam, mgr, dets or [])


class TestModuloAlertas(unittest.TestCase):

    def test_persona_movil_cerca_es_critica_y_recalcula(self):
        f = _frame([VisionDetection(bearing=0.0, label="person",
                                    range_est=0.8, moving=True)])
        als, rec = ModuloAlertas(conPip=False).evaluar(f)
        self.assertTrue(rec)
        self.assertEqual(als[0][0], 0)
        self.assertIn("persona", als[0][1])

    def test_movil_sustituye_la_alerta_generica_del_sector(self):
        f = _frame([VisionDetection(bearing=0.0, label="person",
                                    range_est=0.8, moving=True)])
        als, _ = ModuloAlertas(conPip=False).evaluar(f)
        textos = [t for _p, t in als]
        self.assertFalse(any("person cerca" in t for t in textos))

    def test_hoyo_adelante_avisa_en_pasos(self):
        f = _frame([VisionDetection(bearing=0.1, label="hoyo",
                                    range_est=2.0, hazard=True)])
        als, rec = ModuloAlertas(conPip=False).evaluar(f)
        self.assertTrue(any("hoyo" in t and "pasos" in t for _p, t in als))
        self.assertFalse(rec)          # a 2 m aun no es critica

    def test_hoyo_pegado_es_critico(self):
        f = _frame([VisionDetection(bearing=0.0, label="hoyo",
                                    range_est=1.0, hazard=True)])
        als, rec = ModuloAlertas(conPip=False).evaluar(f)
        self.assertTrue(rec)
        self.assertTrue(any(p == 0 and "hoyo" in t for p, t in als))

    def test_cooldown_por_tipo(self):
        mod = ModuloAlertas(conPip=False)
        f = _frame([VisionDetection(bearing=0.0, label="hoyo",
                                    range_est=2.0, hazard=True)])
        self.assertTrue(mod.evaluar(f)[0])
        self.assertEqual(mod.evaluar(f)[0], [])     # misma alerta, calla

    def test_apagado_y_toggle(self):
        mod = ModuloAlertas(conPip=False)
        f = _frame([VisionDetection(bearing=0.0, label="person",
                                    range_est=0.5, moving=True)])
        self.assertFalse(mod.alternar())            # queda apagado
        self.assertEqual(mod.evaluar(f), ([], False))
        self.assertTrue(mod.alternar())
        self.assertTrue(mod.evaluar(f)[0])

    def test_sin_frame_no_truena(self):
        self.assertEqual(ModuloAlertas(conPip=False).evaluar(None), ([], False))


class TestBoton(unittest.TestCase):

    def test_doble_click_alterna_y_sencillo_no(self):
        cambios = []
        b = BotonPullUp(alCambiar=lambda: cambios.append(1), ventanaS=0.5)
        b.pulsado()
        self.assertEqual(cambios, [])
        b.pulsado()                                 # doble
        self.assertEqual(cambios, [1])
        time.sleep(0.6)
        b.pulsado()                                 # sencillo de nuevo
        self.assertEqual(cambios, [1])


def _pickModel():
    for base in ("models/nav_ppo", "models/nav_ppo_extended_cmp"):
        if os.path.exists(base + "_policy.npz"):
            return base
    return None


def _mapaConPared(bloqueaTodo=False):
    """pasillo libre con una pared enfrente a 3 m (o un encierro)."""
    mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
    for _ in range(3):
        obs = [Observation(range_m=3.0, bearing=b, sigma_r=0.05)
               for b in [k * 0.12 - 0.72 for k in range(13)]]
        mgr.updateFromScan((0.0, 0.0, 0.0), obs)
    for x in range(-20, 80):
        mgr.clearFootprint(x * 0.1, 0.0, radius=0.9)
    if bloqueaTodo:
        for ang in range(0, 720):          # anillo doble bien tupido
            a = math.radians(ang / 2.0)
            r = 1.1 + 0.05 * (ang % 2)
            for _ in range(3):
                mgr.updateFromScan((0.0, 0.0, 0.0),
                                   [Observation(range_m=r, bearing=a,
                                                sigma_r=0.05)])
    return mgr


@unittest.skipUnless(_pickModel(), "falta el modelo entrenado")
class TestPlanificador(unittest.TestCase):

    def setUp(self):
        from navigation.navigator import load_policy
        self.policy, _ = load_policy(_pickModel())

    def test_llega_en_linea_recta(self):
        from navigation.route_planner import planear
        mgr = _mapaConPared()
        r = planear(self.policy, mgr, (0.0, 0.0, 0.0), (2.0, 0.0))
        self.assertEqual(r["estado"], "llega")
        self.assertGreater(len(r["points"]), 3)
        fin = r["points"][-1]
        self.assertLess(math.hypot(fin[0] - 2.0, fin[1]), 0.8)

    def test_encerrado_reporta_bloqueada(self):
        from navigation.route_planner import planear
        mgr = _mapaConPared(bloqueaTodo=True)
        r = planear(self.policy, mgr, (0.0, 0.0, 0.0), (5.0, 0.0))
        self.assertEqual(r["estado"], "bloqueada")

    def test_goal_sobre_pared_se_proyecta_a_celda_libre(self):
        from navigation.route_planner import planear, _goalAlcanzable
        mgr = _mapaConPared()
        goalPared = (3.0, 0.0)       # justo en la pared pintada
        ajustado = _goalAlcanzable(mgr, goalPared)
        gx, gy = mgr.worldToCell(*ajustado)
        self.assertFalse(mgr.isBlockedCell(gx, gy))
        r = planear(self.policy, mgr, (0.0, 0.0, 0.0), goalPared)
        self.assertEqual(r["estado"], "llega")

    def test_hilo_entrega_ruta(self):
        from navigation.route_planner import PlanificadorRuta
        mgr = _mapaConPared()
        pl = PlanificadorRuta(self.policy, cadaS=0.1)
        pl.solicitar(mgr, (0.0, 0.0, 0.0), (2.0, 0.0), urgente=True)
        for _ in range(40):
            if pl.ruta is not None:
                break
            time.sleep(0.1)
        self.assertIsNotNone(pl.ruta)
        self.assertEqual(pl.ruta["estado"], "llega")
        pl.limpiar()
        self.assertIsNone(pl.ruta)


if __name__ == "__main__":
    unittest.main()


class SalidaFalsa:
    CRITICAL, WARNING = 3, 2
    def __init__(self):
        self.pips = []
    def pip(self, nivel, direccion="frente", tipo="obstaculo",
            distanciaM=None):
        self.pips.append((nivel, direccion, tipo))
    def recently_active(self, s=0.5):
        return False


class TestPips(unittest.TestCase):
    """la entrega nueva: precaucion = pip, critica = pip (+voz aparte)."""

    def _mod(self):
        dichas = []
        mod = ModuloAlertas(conPip=False,
                            hablarFn=lambda t: dichas.append(t))
        mod.salida = SalidaFalsa()
        return mod, dichas

    def test_precaucion_suena_warning(self):
        mod, dichas = self._mod()
        f = _frame([VisionDetection(bearing=0.1, label="hoyo",
                                    range_est=2.0, hazard=True)])
        mod.evaluar(f)
        self.assertEqual(mod.salida.pips[0][0], SalidaFalsa.WARNING)
        self.assertEqual(mod.salida.pips[0][1], "frente")

    def test_critica_suena_critical(self):
        mod, _ = self._mod()
        f = _frame([VisionDetection(bearing=0.0, label="person",
                                    range_est=0.5, moving=True)])
        mod.evaluar(f)
        niveles = [p[0] for p in mod.salida.pips]
        self.assertIn(SalidaFalsa.CRITICAL, niveles)

    def test_sin_pips_critica_se_habla(self):
        dichas = []
        mod = ModuloAlertas(conPip=False,
                            hablarFn=lambda t: dichas.append(t))
        f = _frame([VisionDetection(bearing=0.0, label="person",
                                    range_est=0.5, moving=True)])
        mod.evaluar(f)
        self.assertTrue(dichas)          # fallback: voz como antes

    def test_sector_izquierda_panea_izquierda(self):
        mod, _ = self._mod()
        f = _frame([VisionDetection(bearing=math.radians(60),
                                    label="person", range_est=0.8,
                                    moving=True)])
        mod.evaluar(f)
        self.assertEqual(mod.salida.pips[0][1], "izquierda")


class TestUltrasonido(unittest.TestCase):

    def test_reporte_directo_pip_y_pendiente(self):
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        mod.reportarSensor("derecha", 0.6)
        self.assertEqual(mod.salida.pips[0][0], SalidaFalsa.CRITICAL)
        self.assertEqual(mod.salida.pips[0][1], "derecha")
        als, rec = mod.evaluar(_frame([])[0] if False else None)
        # evaluar(None) con activo entrega lo pendiente del sensor
        self.assertTrue(rec)
        self.assertIn("Ultrasonido", als[0][1])
        # y ya no se repite
        self.assertEqual(mod.evaluar(None), ([], False))

    def test_conectar_ultrasonido_mapea_direcciones(self):
        from alertas import conectarUltrasonido
        from interfaces import UltrasonicReading
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        lecturas = [UltrasonicReading(range_m=0.9,
                                      sensor_bearing=math.radians(90)),
                    UltrasonicReading(range_m=0.8,
                                      sensor_bearing=math.radians(-90)),
                    UltrasonicReading(range_m=3.0, sensor_bearing=0.0)]
        conectarUltrasonido(mod, lecturas, umbralM=1.5)
        dirs = [p[1] for p in mod.salida.pips]
        self.assertEqual(dirs, ["izquierda", "derecha"])   # el de 3m no

    def test_apagado_ignora_sensor(self):
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        mod.alternar()
        mod.reportarSensor("frente", 0.5)
        self.assertEqual(mod.salida.pips, [])
