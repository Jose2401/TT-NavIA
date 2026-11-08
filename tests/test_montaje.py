"""
pruebas del montaje fisico (cad en docs/) y de la validacion cruzada
vision <-> ultrasonicos frontales. correr: python -m unittest tests.test_montaje
"""
import math
import unittest

from interfaces import VisionDetection, UltrasonicReading
from montaje import (usPosicion, usAlCentro, bearingEstereo, rangoPiso,
                     esEcoDePiso, US_SEPARACION_M)
from vision_bridge import validarConUltrasonido, fuseObservations


def par(rI, rD, validos=(True, True)):
    (oyI, bI), (oyD, bD) = usPosicion("izq"), usPosicion("der")
    return [UltrasonicReading(rI, bI, 7.65, validos[0], oyI, "izq"),
            UltrasonicReading(rD, bD, 7.65, validos[1], oyD, "der")]


class TestGeometria(unittest.TestCase):

    def test_frontales_paralelos_y_separados(self):
        (oyI, bI), (oyD, bD) = usPosicion("izq"), usPosicion("der")
        self.assertAlmostEqual(bI, 0.0)
        self.assertAlmostEqual(bD, 0.0)
        self.assertAlmostEqual(oyI - oyD, US_SEPARACION_M)
        self.assertGreater(oyI, 0)          # + = izquierda

    def test_al_centro_corrige_el_desplazamiento(self):
        r, b = usAlCentro(2.0, 0.0, 0.05)
        self.assertAlmostEqual(r, math.hypot(2.0, 0.05), places=6)
        self.assertGreater(b, 0)            # sensor izq -> algo a la izq

    def test_estereo_lado_y_zona_muerta(self):
        self.assertGreater(bearingEstereo(1.0, 1.1), 0)    # mas cerca izq
        self.assertLess(bearingEstereo(1.1, 1.0), 0)
        self.assertEqual(bearingEstereo(2.00, 2.03), 0.0)  # ruido = frente

    def test_piso_segun_pitch(self):
        self.assertIsNone(rangoPiso(-20))                  # viendo arriba
        self.assertLess(rangoPiso(30), rangoPiso(10))
        self.assertTrue(esEcoDePiso(2.6, 25))
        self.assertFalse(esEcoDePiso(1.0, 25))
        self.assertFalse(esEcoDePiso(2.6, None))           # sin imu: no adivina


class TestValidacionCruzada(unittest.TestCase):

    def test_puerta_de_vidrio_se_corrige_a_cerrada(self):
        d = VisionDetection(0.0, "door", range_est=2.0, estado="abierta")
        validarConUltrasonido([d], par(2.05, 2.08), pitchDeg=8)
        self.assertEqual(d.estado, "cerrada")
        self.assertEqual(d.confirmado, "ambos")
        self.assertAlmostEqual(d.range_est, 2.05)          # rango preciso

    def test_puerta_abierta_el_eco_pasa_por_el_vano(self):
        d = VisionDetection(0.0, "door", range_est=2.0, estado="cerrada")
        validarConUltrasonido([d], par(5.0, 5.1), pitchDeg=8)
        self.assertEqual(d.estado, "abierta")

    def test_confirmado_sube_confianza_y_usa_rango_us(self):
        d = VisionDetection(0.0, "person", range_est=1.8, confidence=0.6)
        validarConUltrasonido([d], par(1.62, 1.65), pitchDeg=8)
        self.assertEqual(d.confirmado, "ambos")
        self.assertAlmostEqual(d.range_est, 1.62)
        self.assertGreater(d.confidence, 0.6)

    def test_objeto_bajo_el_haz_no_se_contradice(self):
        d = VisionDetection(0.0, "chair", range_est=1.0)
        validarConUltrasonido([d], par(4.0, 4.1), pitchDeg=8)
        self.assertEqual(d.confirmado, "")

    def test_objeto_alto_sin_eco_queda_contradicho(self):
        d = VisionDetection(0.0, "fridge", range_est=2.0)
        validarConUltrasonido([d], par(5.5, 5.6), pitchDeg=8)
        self.assertEqual(d.confirmado, "contradicho")

    def test_fuera_del_cono_no_se_toca(self):
        d = VisionDetection(math.radians(60), "fridge", range_est=1.5)
        validarConUltrasonido([d], par(5.0, 5.0), pitchDeg=8)
        self.assertEqual(d.confirmado, "")

    def test_moviles_y_peligros_no_se_validan(self):
        p = VisionDetection(0.0, "person", range_est=1.0, moving=True)
        h = VisionDetection(0.0, "hoyo", range_est=1.0, hazard=True)
        validarConUltrasonido([p, h], par(5.0, 5.0), pitchDeg=8)
        self.assertEqual((p.confirmado, h.confirmado), ("", ""))

    def test_eco_de_piso_queda_como_corredor_libre(self):
        limpias = validarConUltrasonido([], par(2.6, 2.6), pitchDeg=25)
        self.assertTrue(all(not u.valid for u in limpias))
        self.assertLess(limpias[0].max_range, 2.6)         # horizontal

    def test_sin_ultrasonicos_no_cambia_nada(self):
        d = VisionDetection(0.0, "fridge", range_est=2.0)
        validarConUltrasonido([d], [], pitchDeg=8)
        self.assertEqual(d.confirmado, "")

    def test_fusion_usa_polar_desde_el_centro(self):
        d = VisionDetection(0.0, "person", range_est=1.7)
        obs = fuseObservations([d], par(1.62, 1.65))
        persona = [o for o in obs if o.label == "person"][0]
        self.assertLess(persona.sigma_r, 0.1)              # rango del us


class TestAlertasCruzadas(unittest.TestCase):

    def _frame(self, dets, lect):
        from ekf_slam import EKFSlam
        from map_manager import ChunkedMapManager
        from navigation_feed import buildNavigationFrame
        return buildNavigationFrame(EKFSlam(0, 0, 0),
                                    ChunkedMapManager(mapDir=None),
                                    dets, ultrasonic=lect)

    def test_eco_sin_vision_es_pared_al_frente(self):
        from alertas import ModuloAlertas
        f = self._frame([], par(0.6, 0.62))
        als, rec = ModuloAlertas(conPip=False).evaluar(f)
        self.assertTrue(rec)
        self.assertTrue(any(p == 0 and "pared" in t for p, t in als))

    def test_contradicha_no_grita(self):
        from alertas import ModuloAlertas
        d = VisionDetection(0.0, "fridge", range_est=0.7,
                            confirmado="contradicho")
        f = self._frame([d], par(5.0, 5.0))
        als, rec = ModuloAlertas(conPip=False).evaluar(f)
        self.assertFalse(any(p == 0 for p, _t in als))
        self.assertTrue(any("Posible" in t for _p, t in als))

    def test_confirmada_no_duplica_con_el_eco(self):
        from alertas import ModuloAlertas
        d = VisionDetection(0.0, "person", range_est=0.7,
                            confirmado="ambos")
        f = self._frame([d], par(0.7, 0.71))
        als, _ = ModuloAlertas(conPip=False).evaluar(f)
        self.assertFalse(any("pared" in t for _p, t in als))


class ImuFalsa:
    """simula una placa atornillada donde el cabeceo sale en ROLL e
    invertido (el caso feo), pa ver que la calibracion lo descubre."""
    placement = None

    def __init__(self):
        self.fase = 0

    def readEulerCrudo(self):
        return {0: (100.0, 3.0, -1.0),      # derecho
                1: (100.0, -27.0, -1.5),    # agachado 30: roll baja
                2: (55.0, 3.0, -1.0)}[self.fase]   # giro izq: heading baja


class TestCalibracionImu(unittest.TestCase):

    def test_descubre_campo_signos_y_cero(self):
        import os, tempfile
        import odometry
        imu = ImuFalsa()
        orig = odometry._promedio
        odometry._promedio = lambda i, segundos=1.5: i.readEulerCrudo()
        pasos = iter([0, 1, 2])
        def entrada(_msg):
            imu.fase = next(pasos)
        ruta = os.path.join(tempfile.mkdtemp(), "m.json")
        try:
            m = odometry.calibrarMontaje(imu, ruta=ruta, entrada=entrada)
        finally:
            odometry._promedio = orig
        self.assertEqual(m["pitchDesde"], "roll")
        self.assertEqual(m["pitchSigno"], -1.0)
        self.assertEqual(m["headingSigno"], -1.0)
        # con el json cargado, cabeza derecha da ~0 y agachado da ~+30
        b = odometry.Bno055.__new__(odometry.Bno055)
        b._cargarMontaje(ruta)
        b.readEulerCrudo = lambda: (100.0, 3.0, -1.0)
        self.assertAlmostEqual(math.degrees(b.readEuler()[2]), 0.0, places=4)
        b.readEulerCrudo = lambda: (100.0, -27.0, -1.5)
        self.assertAlmostEqual(math.degrees(b.readEuler()[2]), 30.0, places=4)


if __name__ == "__main__":
    unittest.main()


class TestDatasheetMB1232(unittest.TestCase):
    """caracteristicas del ez3 segun el datasheet pd11848."""

    def test_alcance_por_tamano_y_voltaje(self):
        from montaje import alcanceUS
        self.assertLess(alcanceUS("bottle"), alcanceUS("person"))
        self.assertLess(alcanceUS("person"), alcanceUS("chair"))
        self.assertLess(alcanceUS("chair"), alcanceUS("door"))
        self.assertLess(alcanceUS("door", volts=3.3),
                        alcanceUS("door", volts=5.0))

    def test_objeto_chico_lejos_no_se_contradice(self):
        # una botella a 2 m: el ez3 "filtra lo chico", no ver eco no
        # significa que la camara se equivoco
        d = VisionDetection(0.0, "bottle", range_est=2.0)
        validarConUltrasonido([d], par(5.0, 5.0), pitchDeg=-30)
        self.assertEqual(d.confirmado, "")

    def test_persona_fuera_de_alcance_a_33v_no_se_contradice(self):
        d = VisionDetection(0.0, "person", range_est=2.3)
        validarConUltrasonido([d], par(6.0, 6.0), pitchDeg=0)
        self.assertEqual(d.confirmado, "")

    def test_20cm_es_lo_mas_cerca_y_si_pita(self):
        from alertas import ModuloAlertas, conectarUltrasonido
        mod = ModuloAlertas(conPip=False)

        class Salida:
            CRITICAL, WARNING = 3, 2
            pips = []
            def pip(self, nivel, *a, **k):
                self.pips.append(nivel)
        mod.salida = Salida()
        conectarUltrasonido(mod, par(0.20, 0.20))
        self.assertEqual(mod.salida.pips, [3])
