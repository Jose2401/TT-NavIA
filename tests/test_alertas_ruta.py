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

    def _par(self, rI, rD, validos=(True, True)):
        from interfaces import UltrasonicReading
        from montaje import usPosicion
        (oyI, bI), (oyD, bD) = usPosicion("izq"), usPosicion("der")
        return [UltrasonicReading(rI, bI, 7.65, validos[0], oyI, "izq"),
                UltrasonicReading(rD, bD, 7.65, validos[1], oyD, "der")]

    def test_frontales_parejos_son_pared_al_frente(self):
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(0.9, 0.92))
        self.assertEqual(mod.salida.pips[0][1], "frente")
        self.assertEqual(mod.salida.pips[0][2], "pared")

    def test_solo_uno_cerca_se_carga_a_su_lado(self):
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(0.7, 3.0))
        self.assertEqual(mod.salida.pips[0][1], "izquierda")
        mod2 = ModuloAlertas(conPip=False)
        mod2.salida = SalidaFalsa()
        conectarUltrasonido(mod2, self._par(3.0, 0.7))
        self.assertEqual(mod2.salida.pips[0][1], "derecha")

    def test_lejos_no_pita(self):
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(3.0, 3.1), umbralM=1.5)
        self.assertEqual(mod.salida.pips, [])

    def test_eco_de_piso_con_cabeza_agachada_no_pita(self):
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        # agachado 40 grados el haz pega al piso a ~1.9 m: eco a 1.8 = piso
        conectarUltrasonido(mod, self._par(1.80, 1.81), umbralM=2.0,
                            pitchDeg=40)
        self.assertEqual(mod.salida.pips, [])
        # lo que esta ANTES del piso si es obstaculo real
        conectarUltrasonido(mod, self._par(1.20, 1.21), umbralM=2.0,
                            pitchDeg=40)
        self.assertEqual(len(mod.salida.pips), 1)
        # cabeza derecha: el mismo 1.8 ya no puede ser piso
        conectarUltrasonido(mod, self._par(1.80, 1.81), umbralM=2.0,
                            pitchDeg=0)
        self.assertEqual(len(mod.salida.pips), 2)

    def test_apagado_ignora_sensor(self):
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        mod.alternar()
        mod.reportarSensor("frente", 0.5)
        self.assertEqual(mod.salida.pips, [])


class TestHallazgosDoc(unittest.TestCase):
    """hallazgos de las pruebas del documento: alertas por mapa
    desalineado y dosificacion de la misma pared."""

    def _frame(self, conf, dets=None, lect=None, dLeft=0.4):
        from dataclasses import replace
        f = _frame(dets or [])
        f = replace(f, d_left=dLeft, ultrasonic=lect or [])
        return replace(f, slam=replace(f.slam, confidence=conf))

    def test_pose_poco_confiable_el_mapa_no_dispara(self):
        mod = ModuloAlertas(conPip=False)
        als, rec = mod.evaluar(self._frame(conf=0.2))
        self.assertEqual(als, [])          # la "pared" a 40 cm era del mapa
        self.assertFalse(rec)

    def test_pose_confiable_el_mapa_si_dispara(self):
        mod = ModuloAlertas(conPip=False)
        als, rec = mod.evaluar(self._frame(conf=0.9))
        self.assertTrue(rec)

    def test_sensor_en_vivo_alerta_aunque_la_pose_sea_mala(self):
        mod = ModuloAlertas(conPip=False)
        det = VisionDetection(bearing=math.radians(60), label="chair",
                              range_est=0.3)
        als, rec = mod.evaluar(self._frame(conf=0.1, dets=[det],
                                           dLeft=float("inf")))
        self.assertTrue(rec)

    def test_misma_pared_suena_una_vez(self):
        from interfaces import UltrasonicReading
        from montaje import usPosicion
        (oyI, bI), (oyD, bD) = usPosicion("izq"), usPosicion("der")
        lect = [UltrasonicReading(1.2, bI, 7.65, True, oyI, "izq"),
                UltrasonicReading(1.21, bD, 7.65, True, oyD, "der")]
        from dataclasses import replace
        f = _frame([])
        f = replace(f, d_front=1.25, ultrasonic=lect)   # mapa + eco: misma pared
        f = replace(f, slam=replace(f.slam, confidence=0.9))
        als, _ = ModuloAlertas(conPip=False).evaluar(f)
        frente = [t for p, t in als if "frente" in t]
        self.assertEqual(len(frente), 1, als)
