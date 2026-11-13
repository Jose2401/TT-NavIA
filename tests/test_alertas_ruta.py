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
        mod.reportarSensor("derecha", 0.22)
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
        conectarUltrasonido(mod, self._par(0.5, 0.52))
        self.assertEqual(mod.salida.pips[0][1], "frente")
        self.assertEqual(mod.salida.pips[0][2], "pared")

    def test_solo_uno_cerca_se_carga_a_su_lado(self):
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(0.5, 3.0))
        self.assertEqual(mod.salida.pips[0][1], "izquierda")
        mod2 = ModuloAlertas(conPip=False)
        mod2.salida = SalidaFalsa()
        conectarUltrasonido(mod2, self._par(3.0, 0.5))
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
        # cabeza derecha: el mismo 1.8 ya no puede ser piso (modulo
        # nuevo: el de arriba ya aviso de ese lado y esta en espera)
        mod2 = ModuloAlertas(conPip=False)
        mod2.salida = SalidaFalsa()
        conectarUltrasonido(mod2, self._par(1.80, 1.81), umbralM=2.0,
                            pitchDeg=0)
        self.assertEqual(len(mod2.salida.pips), 1)

    def test_misma_cosa_quieta_no_pita_en_rafaga(self):
        """el hilo del sensor reporta 10 veces por segundo: la misma
        pared a la misma distancia suena una vez y luego espera."""
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        for _ in range(20):
            conectarUltrasonido(mod, self._par(0.50, 0.51))
        self.assertEqual(len(mod.salida.pips), 1)
        # si se acerca hasta ser critica (otro nivel), suena de nuevo
        conectarUltrasonido(mod, self._par(0.22, 0.23))
        self.assertEqual(len(mod.salida.pips), 2)
        self.assertEqual(mod.salida.pips[1][0], SalidaFalsa.CRITICAL)

    def test_umbrales_de_lo_que_suena(self):
        """pedido del equipo: voz solo a 25 cm o menos; de ahi a 60 cm
        nomas el pip de precaucion; mas lejos nada."""
        from alertas import conectarUltrasonido
        dichas = []
        mod = ModuloAlertas(conPip=False, hablarFn=dichas.append)
        self.assertEqual((mod.criticaM, mod.precaucionM), (0.25, 0.60))
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(0.90, 0.91))
        self.assertEqual(mod.salida.pips, [])
        conectarUltrasonido(mod, self._par(0.45, 0.46))
        self.assertEqual([p[0] for p in mod.salida.pips],
                         [SalidaFalsa.WARNING])
        conectarUltrasonido(mod, self._par(0.24, 0.25))
        self.assertEqual(mod.salida.pips[-1][0], SalidaFalsa.CRITICAL)
        # se pueden mover con los flags de los mains
        doc = ModuloAlertas(conPip=False, criticaM=0.8, precaucionM=1.5)
        doc.salida = SalidaFalsa()
        conectarUltrasonido(doc, self._par(0.70, 0.71))
        self.assertEqual(doc.salida.pips[0][0], SalidaFalsa.CRITICAL)

    def test_navegador_recalcula_aunque_no_suene(self):
        """bajarle a lo que se oye no hace mas lento al navegador: a
        70 cm no suena nada pero si se le pide decidir ya."""
        from dataclasses import replace
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        f = _frame([])
        f = replace(f, d_front=0.70,
                    slam=replace(f.slam, confidence=0.9))
        als, rec = mod.evaluar(f)
        self.assertEqual((als, mod.salida.pips), ([], []))
        self.assertTrue(rec)
        self.assertFalse(mod.evaluar(f)[1])        # dosificado: no cada ciclo

    def test_lo_mismo_quieto_no_pita_por_cambiar_de_lado(self):
        """parado frente a algo, los dos sensores se turnan quien lo ve
        mas cerca y la alerta brincaba de 'frente' a 'izquierda'."""
        mod = ModuloAlertas(conPip=False)
        self.assertTrue(mod._pasa((1, "frente", "obstaculo"), 1, 100.0, 0.44))
        self.assertFalse(mod._pasa((1, "izquierda", "obstaculo"), 1, 103.0,
                                   0.45))
        # otra cosa (a otra distancia) de ese lado si avisa
        self.assertTrue(mod._pasa((1, "izquierda", "obstaculo"), 1, 104.0,
                                  0.20 + 0.44))
        # una precaucion nunca calla a una critica
        self.assertTrue(mod._pasa((0, "frente", "obstaculo"), 0, 105.0, 0.22))
        # la misma lectura critica brincando de lado si se calla...
        self.assertFalse(mod._pasa((0, "derecha", "obstaculo"), 0, 105.5,
                                   0.23))
        # ...pero otra cosa (a otra distancia) de ese lado no
        self.assertTrue(mod._pasa((0, "derecha", "obstaculo"), 0, 106.0,
                                  0.31))

    def test_parado_junto_a_algo_quieto_se_dice_una_vez(self):
        """medido con una escena quieta de 68 s: 5 alertas habladas
        seguidas de lo mismo. parado ya no se repite; se repite cuando
        la imu vuelve a contar pasos."""
        mod = ModuloAlertas(conPip=False)
        caminando = [False]
        mod.caminandoFn = lambda: caminando[0]
        llave = (0, "derecha", "obstaculo")
        self.assertTrue(mod._pasa(llave, 0, 100.0, 0.21))
        for t in (103.0, 110.0, 140.0, 400.0):
            self.assertFalse(mod._pasa(llave, 0, t, 0.22))
        caminando[0] = True                 # echa a andar pegado a eso
        self.assertTrue(mod._pasa(llave, 0, 401.0, 0.22))
        # y si lo que cambia es la distancia, suena aunque este parado
        caminando[0] = False
        self.assertTrue(mod._pasa(llave, 0, 404.0, 0.60))

    def test_persona_al_lado_no_es_alto(self):
        """quien camina junto al usuario no va a chocar con el: pip de
        precaucion, no 'Alto'. de frente si."""
        lado = _frame([VisionDetection(bearing=math.radians(60),
                                       label="person", range_est=0.7,
                                       moving=True)])
        als, _ = ModuloAlertas(conPip=False).evaluar(lado)
        self.assertEqual([p for p, _t in als], [1])
        frente = _frame([VisionDetection(bearing=0.0, label="person",
                                         range_est=0.7, moving=True)])
        als, rec = ModuloAlertas(conPip=False).evaluar(frente)
        self.assertEqual(als[0][0], 0)
        self.assertIn("Alto", als[0][1])

    def test_distancia_de_camara_que_brinca_no_repite(self):
        mod = ModuloAlertas(conPip=False)
        mod.caminandoFn = lambda: False
        llave = (1, "frente", "movil")
        self.assertTrue(mod._pasa(llave, 1, 100.0, 1.60))
        # la misma persona, que la camara ahora pone a 1.85 m
        self.assertFalse(mod._pasa(llave, 1, 120.0, 1.85))
        # se acerco de verdad
        self.assertTrue(mod._pasa(llave, 1, 121.0, 1.00))

    def test_histeresis_en_el_umbral_de_la_voz(self):
        """el eco de algo quieto brinca unos cm: a 24 y luego a 28 cm
        era 'critica' y 'precaucion' por turnos (doble de avisos)."""
        from alertas import conectarUltrasonido
        mod = ModuloAlertas(conPip=False)
        mod.salida = SalidaFalsa()
        conectarUltrasonido(mod, self._par(0.24, 0.24))
        conectarUltrasonido(mod, self._par(0.28, 0.28))    # sigue critica
        self.assertEqual([p[0] for p in mod.salida.pips],
                         [SalidaFalsa.CRITICAL])
        conectarUltrasonido(mod, self._par(0.50, 0.50))    # ya se alejo
        self.assertEqual(mod.salida.pips[-1][0], SalidaFalsa.WARNING)
        self.assertFalse(mod._enCritica)

    def test_alertas_del_mapa_traen_su_distancia(self):
        from dataclasses import replace
        from navigation_feed import alertasDeObstaculo, safetyAlerts
        f = _frame([])
        f = replace(f, d_front=0.5, slam=replace(f.slam, confidence=0.9))
        det = alertasDeObstaculo(f, 0.25, 0.60)
        self.assertEqual([(p, sec, d) for p, _t, sec, d in det],
                         [(1, "front", 0.5)])
        self.assertEqual(safetyAlerts(f, 0.25, 0.60),
                         [(p, t) for p, t, _s, _d in det])
        # con la distancia, el mapa tambien deja de repetir lo quieto
        mod = ModuloAlertas(conPip=False)
        mod.caminandoFn = lambda: False
        self.assertTrue(mod.evaluar(f)[0])
        mod._ultimo = {k: (t - 60.0, d, e)
                       for k, (t, d, e) in mod._ultimo.items()}
        self.assertEqual(mod.evaluar(f)[0], [])

    def test_quieto_va_alargando_la_espera(self):
        mod = ModuloAlertas(conPip=False)
        llave = (0, "frente", "obstaculo")
        self.assertTrue(mod._pasa(llave, 0, 100.0, 0.90))
        self.assertFalse(mod._pasa(llave, 0, 101.0, 0.90))
        self.assertTrue(mod._pasa(llave, 0, 102.6, 0.91))    # 2.5 s
        self.assertFalse(mod._pasa(llave, 0, 105.2, 0.90))   # ahora 5 s
        self.assertTrue(mod._pasa(llave, 0, 107.7, 0.90))
        # se acerco de verdad: vuelve la espera normal
        self.assertTrue(mod._pasa(llave, 0, 110.3, 0.50))

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
        lect = [UltrasonicReading(0.5, bI, 7.65, True, oyI, "izq"),
                UltrasonicReading(0.51, bD, 7.65, True, oyD, "der")]
        from dataclasses import replace
        f = _frame([])
        f = replace(f, d_front=0.55, ultrasonic=lect)   # mapa + eco: misma pared
        f = replace(f, slam=replace(f.slam, confidence=0.9))
        als, _ = ModuloAlertas(conPip=False).evaluar(f)
        frente = [t for p, t in als if "frente" in t]
        self.assertEqual(len(frente), 1, als)
