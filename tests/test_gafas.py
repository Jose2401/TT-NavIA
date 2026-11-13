"""
pruebas del hardware de las gafas SIN hardware: el lector del rp2040
(sensores.py) se prueba dandole lineas de texto como las del firmware,
el boton sin gpio y el oido sin microfono.
correr: python -m unittest tests.test_gafas
"""
import math
import queue
import time
import unittest

from interfaces import UltrasonicReading, Observation
from odometry import StepDetector, MotionEstimator
from sensores import SensoresGafas, ImuGafas, UltrasonicosGafas


def _imu(ms, h=0.0, r=0.0, p=0.0, w=(0, 0, 0), lin=(0, 0, 0),
         grav=(0, 0, 980), cal=0xFF):
    """una linea I como la manda firmware/code.py (h, r, p en grados)."""
    v = [ms, int(h * 16), int(r * 16), int(p * 16),
         int(w[0] * 16), int(w[1] * 16), int(w[2] * 16),
         int(lin[0] * 100), int(lin[1] * 100), int(lin[2] * 100),
         grav[0], grav[1], grav[2], cal]
    return "I," + ",".join(str(x) for x in v)


class TestLectorRP2040(unittest.TestCase):

    def setUp(self):
        self.s = SensoresGafas(iniciar=False)

    def test_linea_imu(self):
        self.s.procesarLinea(_imu(1000, h=90.0, p=12.5, cal=0b11100111))
        self.assertTrue(self.s.imuOk)
        self.assertAlmostEqual(self.s.euler[0], 90.0)
        self.assertAlmostEqual(self.s.euler[2], 12.5)
        self.assertEqual(self.s.calib, {"sys": 3, "gyro": 2, "accel": 1,
                                        "mag": 3})

    def test_basura_no_truena(self):
        for linea in ("", "hola", "I,1,2,3", "U,xx,D,yy", "I," + "a," * 14,
                      "\x1b]0;code.py\x1b\\", "E,imu sin respuesta"):
            self.s.procesarLinea(linea)
        self.assertFalse(self.s.imuOk)
        self.assertEqual(len(self.s.errores), 1)

    def test_titulo_de_circuitpython_pegado_a_la_linea(self):
        self.s.procesarLinea("\x1b]0;\U0001f40dcode.py | 10.3.1\x1b\\"
                             + _imu(5, h=10.0))
        self.assertTrue(self.s.imuOk)

    def test_giro_acumulado_y_se_vacia_al_tomarlo(self):
        # 10 muestras girando 2 grados cada una (el giroscopio coincide)
        for k in range(11):
            self.s.procesarLinea(_imu(1000 + 20 * k, h=2.0 * k,
                                      w=(0, 0, -100)))
        giro, pasos = self.s.tomarMovimiento()
        self.assertAlmostEqual(giro, 20.0, places=3)
        self.assertEqual(pasos, 0)
        self.assertEqual(self.s.tomarMovimiento(), (0.0, 0))

    def test_cruce_de_360(self):
        self.s.procesarLinea(_imu(0, h=359.0, w=(0, 0, -100)))
        self.s.procesarLinea(_imu(20, h=1.0, w=(0, 0, -100)))
        self.assertAlmostEqual(self.s.tomarMovimiento()[0], 2.0, places=3)

    def test_brinco_del_magnetometro_se_ignora(self):
        """el heading salta 40 grados pero el giroscopio dice que la
        cabeza no se movio: no es un giro del usuario."""
        self.s.procesarLinea(_imu(0, h=100.0))
        self.s.procesarLinea(_imu(20, h=100.0))
        self.s.procesarLinea(_imu(40, h=140.0))          # brinco
        self.s.procesarLinea(_imu(60, h=140.0))
        giro, _ = self.s.tomarMovimiento()
        self.assertAlmostEqual(giro, 0.0, places=2)
        self.assertEqual(self.s.brincos, 1)

    def test_giro_rapido_real_no_es_brinco(self):
        # 300 grados/s: 6 grados por muestra, y el giroscopio lo confirma
        for k in range(6):
            self.s.procesarLinea(_imu(20 * k, h=6.0 * k, w=(0, 0, -300)))
        self.assertAlmostEqual(self.s.tomarMovimiento()[0], 30.0, places=2)
        self.assertEqual(self.s.brincos, 0)

    def test_pasos_a_la_frecuencia_de_la_imu(self):
        """caminar = picos verticales cada 0.5 s; se cuentan en el hilo
        (50 Hz) aunque el ciclo de vision los recoja cada medio segundo."""
        ms = 0
        for paso in range(6):
            for k in range(25):                    # 0.5 s por paso
                az = 3.0 if 3 <= k < 7 else 0.1
                self.s.procesarLinea(_imu(ms, lin=(0, 0, az)))
                ms += 20
        _g, pasos = self.s.tomarMovimiento()
        self.assertEqual(pasos, 6)

    def test_sacudida_horizontal_no_es_paso(self):
        # voltear rapido la cabeza acelera de lado, no hacia arriba
        ms = 0
        for _ in range(4):
            for k in range(25):
                ax = 4.0 if 3 <= k < 7 else 0.0
                self.s.procesarLinea(_imu(ms, lin=(ax, 0, 0)))
                ms += 20
        self.assertEqual(self.s.tomarMovimiento()[1], 0)

    def test_ultrasonicos(self):
        ahora = time.time()
        self.s.procesarLinea("U,100,I,84", ahora)
        self.s.procesarLinea("U,200,D,765", ahora)
        lect = {u.lado: u for u in UltrasonicosGafas(self.s).read()}
        self.assertAlmostEqual(lect["izq"].range_m, 0.84)
        self.assertTrue(lect["izq"].valid)
        self.assertGreater(lect["izq"].offset_y, 0)      # izq = y positiva
        self.assertFalse(lect["der"].valid)              # 765 = sin eco

    def test_20cm_es_valido(self):
        self.s.procesarLinea("U,1,D,20", time.time())
        u = self.s.lecturasUS()[0]
        self.assertTrue(u.valid)
        self.assertAlmostEqual(u.range_m, 0.20)

    def test_lectura_vieja_o_ausente_no_sale(self):
        self.s.procesarLinea("U,1,I,90", time.time() - 5.0)
        self.assertEqual(self.s.lecturasUS(), [])
        self.s.procesarLinea("U,2,I,90", time.time())
        self.s.procesarLinea("U,3,I,-1", time.time())    # dejo de responder
        self.assertEqual(self.s.lecturasUS(), [])

    def test_eco_nuevo_llama_al_callback(self):
        vistos = []
        self.s.alEco = lambda lect: vistos.append(lect)
        self.s.procesarLinea("U,1,I,60", time.time())
        self.assertEqual(len(vistos), 1)
        self.assertAlmostEqual(vistos[0][0].range_m, 0.60)


class TestImuGafas(unittest.TestCase):

    def test_convencion_del_sistema(self):
        """heading del bno055 sube a la derecha -> en el sistema girar a
        la derecha es negativo; pitch + = viendo al piso."""
        s = SensoresGafas(iniciar=False)
        imu = ImuGafas(s, montajePath="/no/existe.json")
        for k in range(6):
            s.procesarLinea(_imu(20 * k, h=3.0 * k, p=20.0,
                                 w=(0, 0, -150)))
        giro, pasos, pitch = imu.tomarMovimiento()
        self.assertAlmostEqual(math.degrees(giro), -15.0, places=2)
        self.assertAlmostEqual(pitch, 20.0, places=3)

    def test_estimador_usa_el_movimiento_acumulado(self):
        s = SensoresGafas(iniciar=False)
        imu = ImuGafas(s, montajePath="/no/existe.json")
        s.procesarLinea(_imu(0))
        est = MotionEstimator(imu=imu, stepLengthM=0.5)
        est.update(None, 0.3)                      # primer ciclo: se descarta
        ms = 20
        for paso in range(4):
            for k in range(25):
                az = 3.0 if 3 <= k < 7 else 0.1
                s.procesarLinea(_imu(ms, h=0.4 * (ms / 20), lin=(0, 0, az),
                                     w=(0, 0, -20)))
                ms += 20
        m, info = est.update(None, 2.0)
        self.assertAlmostEqual(m.delta_trans, 4 * 0.5)
        self.assertAlmostEqual(math.degrees(m.delta_rot), -40.0, places=1)
        self.assertEqual(info["source"], "imu")
        self.assertEqual(info["steps_total"], 4)

    def test_sin_datos_cae_al_respaldo(self):
        s = SensoresGafas(iniciar=False)
        imu = ImuGafas(s, montajePath="/no/existe.json")
        self.assertFalse(imu.available)
        self.assertIsNone(imu.tomarMovimiento())


class TestPasosConfirmados(unittest.TestCase):

    def test_pico_suelto_no_cuenta(self):
        det = StepDetector(confirmar=True)
        n = sum(int(det.update(3.0 if 10 <= k < 14 else 0.1, k * 0.02))
                for k in range(200))
        self.assertEqual(n, 0)

    def test_al_confirmar_cuentan_los_dos(self):
        det = StepDetector(confirmar=True)
        n = 0
        for k in range(100):                       # picos en 0.2 s y 0.7 s
            a = 3.0 if k in (10, 11, 35, 36) else 0.1
            n += int(det.update(a, k * 0.02))
        self.assertEqual(n, 2)
        self.assertEqual(det.stepsTotal, 2)


class TestRayosLibres(unittest.TestCase):

    def _par(self, vI, vD, maxI=7.65, maxD=7.65):
        return [UltrasonicReading(7.65 if not vI else 1.0, 0.0, maxI, vI,
                                  0.05, "izq"),
                UltrasonicReading(7.65 if not vD else 1.0, 0.0, maxD, vD,
                                  -0.05, "der")]

    def test_sin_limite_como_siempre(self):
        from vision_bridge import freeSpaceRays
        rayos = freeSpaceRays(self._par(False, False))
        self.assertEqual(len(rayos), 2)
        self.assertGreater(rayos[0].range_m, 7.6)

    def test_con_limite_solo_si_ninguno_oyo_nada(self):
        from vision_bridge import freeSpaceRays
        self.assertEqual(
            len(freeSpaceRays(self._par(False, False), maxLibreM=2.0)), 2)
        self.assertEqual(
            freeSpaceRays(self._par(False, False), maxLibreM=2.0)[0].range_m,
            2.0)
        # uno si oyo algo: el "sin eco" del otro no despeja nada
        self.assertEqual(freeSpaceRays(self._par(True, False),
                                       maxLibreM=2.0), [])

    def test_recorte_por_piso_si_vale_solo(self):
        from vision_bridge import freeSpaceRays
        rayos = freeSpaceRays(self._par(True, False, maxD=1.4),
                              maxLibreM=2.0)
        self.assertEqual(len(rayos), 1)
        self.assertAlmostEqual(rayos[0].range_m, 1.4)

    def test_rayo_libre_corto_no_pinta_ocupado(self):
        from map_manager import ChunkedMapManager
        from occupancy_grid import OccupancyGrid
        libre = [Observation(range_m=2.0, bearing=0.0, label="libre")]
        mgr = ChunkedMapManager(mapDir=None, chunkSizeM=8.0)
        grid = OccupancyGrid()
        for _ in range(5):
            mgr.updateFromScan((0.0, 0.0, 0.0), libre)
            grid.updateFromScan((0.0, 0.0, 0.0), libre)
        self.assertGreaterEqual(mgr.castDistance(0.0, 0.0, 0.0, 4.0), 3.9)
        gx, gy = grid.worldToGrid(2.0, 0.0)
        self.assertFalse(grid.isOccupied(gx, gy))


class TestBotonClicks(unittest.TestCase):

    def test_un_click_escucha_dos_alertas(self):
        from alertas import BotonPullUp
        clicks, dobles = [], []
        b = BotonPullUp(alCambiar=lambda: dobles.append(1),
                        alClick=lambda: clicks.append(1), ventanaS=0.15,
                        usarGpio=False)
        b.pulsado()
        time.sleep(0.3)
        self.assertEqual((clicks, dobles), ([1], []))
        b.pulsado()
        time.sleep(0.05)
        b.pulsado()                                  # doble
        time.sleep(0.3)
        self.assertEqual((clicks, dobles), ([1], [1]))
        # tres seguidos: un doble y luego empieza otro sencillo
        b.pulsado(); b.pulsado(); b.pulsado()
        time.sleep(0.3)
        self.assertEqual((clicks, dobles), ([1, 1], [1, 1]))

    def test_sin_gpio_no_truena(self):
        from alertas import BotonPullUp
        b = BotonPullUp(usarGpio=False)
        self.assertFalse(b.disponible)
        b.led(True); b.parpadear(); b.pulsado(); b.cerrar()


class TestOidoBoton(unittest.TestCase):

    def _oido(self, modo):
        from interaccion import OidoVoz
        oido = OidoVoz.__new__(OidoVoz)              # sin vosk ni microfono
        oido.modo, oido.autoApagarS = modo, 30.0
        oido.alCambiar = None
        oido._escuchando = modo != "boton"
        oido._tVoz, oido._sucio = time.time(), False
        oido.alOir = oido.ecoFn = None
        oido._sAudio, oido._tramosEco = 0.0, []
        oido._tParcial = oido._tAbierto = 0.0
        oido._oyoAlgo = False
        oido.hayMic, oido.medioDuplex, oido._ecoOido = True, False, 0
        oido._cola = queue.Queue()
        import threading
        oido._lock = threading.Lock()

        class Kaldi:
            def __init__(self):
                self.bloques = 0
            def AcceptWaveform(self, d):
                self.bloques += 1
                return self.bloques % 3 == 0
            def Result(self):
                return '{"text": "pausa"}'
            def FinalResult(self):
                return '{"text": "continua"}'
            def PartialResult(self):
                return '{"partial" : ""}'
            def Reset(self):
                self.bloques = 0
        oido._kaldi = Kaldi()
        return oido

    def test_en_modo_boton_no_oye_hasta_el_click(self):
        oido = self._oido("boton")
        for _ in range(6):
            oido.alimentar(b"\0" * 4000)
        self.assertTrue(oido._cola.empty())
        cambios = []
        oido.alCambiar = cambios.append
        self.assertTrue(oido.alternar())
        for _ in range(3):
            oido.alimentar(b"\0" * 4000)
        self.assertEqual(oido._cola.get_nowait(), "pausa")
        self.assertEqual(cambios, [True])

    def test_al_soltar_entrega_lo_que_iba_diciendo(self):
        oido = self._oido("boton")
        oido.alternar()
        oido.alimentar(b"\0" * 4000)                 # a media frase
        self.assertFalse(oido.alternar())
        self.assertEqual(oido._cola.get_nowait(), "continua")

    def test_mientras_habla_el_tts_no_oye(self):
        oido = self._oido("continuo")
        for _ in range(6):
            oido.alimentar(b"\0" * 4000, mudo=True)
        self.assertTrue(oido._cola.empty())

    def test_se_apaga_solo_tras_el_silencio(self):
        oido = self._oido("boton")
        oido.alternar()
        oido._tVoz = time.time() - 60.0
        oido.alimentar(b"\0" * 4000)
        self.assertFalse(oido.escuchando)


class TestAlertasPorVoz(unittest.TestCase):

    def test_prender_y_apagar_las_alertas_hablando(self):
        from interaccion import Interaccion

        class Nav:
            pass
        estados = []
        inter = Interaccion(Nav())
        inter.alAlertas = estados.append
        m = inter.manejarTexto("desactiva las alertas")
        self.assertEqual(m[0].text, "Alertas desactivadas.")
        m = inter.manejarTexto("por favor activa los avisos")
        self.assertEqual(m[0].text, "Alertas activadas.")
        self.assertEqual(estados, [False, True])

    def test_modulo_activar(self):
        from alertas import ModuloAlertas
        mod = ModuloAlertas(conPip=False)
        self.assertFalse(mod.activar(False))
        self.assertEqual(mod.evaluar(None), ([], False))
        self.assertTrue(mod.activar(True))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------
# lo ajustado con la primera caminata real (2026-11-10): pasos lentos,
# distancia por cadencia, comprobacion por ultrasonido
# ---------------------------------------------------------------------
def _caminata(sens, pasos=10, cadaS=0.8, pico=2.4, ms0=0, eco=None,
              relojReal=False):
    """lineas como las de una caminata real con las gafas: cada paso un
    pico vertical de ~0.1 s y luego un valle. eco(t) -> cm o None.
    relojReal: las muestras llegan "ahorita" (pa que la imu cuente como
    viva); si no, con el tiempo de la grabacion."""
    ms = ms0
    n = int(cadaS / 0.02)
    for paso in range(pasos):
        for k in range(n):
            az = pico if 2 <= k < 7 else (-1.0 if 7 <= k < 14 else -0.2)
            sens.procesarLinea(_imu(ms, lin=(0, 0, az)),
                               None if relojReal else ms / 1000.0)
            if eco is not None and k % 10 == 0:
                cm = eco(ms / 1000.0)
                if cm is not None:
                    sens.procesarLinea(f"U,{ms},{'ID'[(k // 10) % 2]},{int(cm)}",
                                       ms / 1000.0)
            ms += 20
    return ms


class _SinCalibracion(unittest.TestCase):
    """las cuentas de estas pruebas son con el paso promedio: se quita
    la calibracion de quien tenga los lentes (calibracion_usuario.json)."""

    def setUp(self):
        import montaje
        self._antes = (montaje.LARGO_PASO_M, montaje.PASO_CALIBRADO)
        montaje.LARGO_PASO_M, montaje.PASO_CALIBRADO = 0.68, False

    def tearDown(self):
        import montaje
        montaje.LARGO_PASO_M, montaje.PASO_CALIBRADO = self._antes


class TestPasosReales(_SinCalibracion):

    def test_paso_lento_y_suave_si_se_cuenta(self):
        """a 75 pasos/min los picos fueron de 1.3 a 2.9 m/s2: con el
        umbral viejo (1.8) se perdia la mitad."""
        s = SensoresGafas(iniciar=False)
        _caminata(s, pasos=10, cadaS=0.8, pico=1.4)
        self.assertEqual(s.tomarMovimiento()[1], 10)
        viejo = SensoresGafas(iniciar=False,
                              stepDetector=StepDetector(confirmar=True))
        _caminata(viejo, pasos=10, cadaS=0.8, pico=1.4)
        self.assertEqual(viejo.tomarMovimiento()[1], 0)

    def test_parado_y_volteando_no_cuenta(self):
        s = SensoresGafas(iniciar=False)
        _caminata(s, pasos=6, cadaS=0.9, pico=0.9)      # lo mas que dio parado
        self.assertEqual(s.tomarMovimiento()[1], 0)

    def test_cadencia_y_largo_de_paso(self):
        import montaje
        s = SensoresGafas(iniciar=False)
        _caminata(s, pasos=10, cadaS=0.8)
        self.assertAlmostEqual(s.pasos.cadencia, 75.0, delta=1.0)
        s.tomarMovimiento()
        # despacio el paso es mas corto que el nominal, en proporcion
        esperado = 10 * montaje.largoPaso(75.0)
        self.assertAlmostEqual(s.avanceM, esperado, delta=0.05)
        self.assertLess(s.avanceM, 10 * montaje.LARGO_PASO_M * 0.8)
        self.assertEqual(s.tomarMovimiento(), (0.0, 0))
        self.assertEqual(s.avanceM, 0.0)

    def test_largo_de_paso_se_recorta(self):
        import montaje
        nominal = montaje.largoPaso(montaje.CADENCIA_PASO_PPM)
        self.assertAlmostEqual(nominal, montaje.LARGO_PASO_M)
        self.assertEqual(montaje.largoPaso(None), montaje.LARGO_PASO_M)
        self.assertAlmostEqual(montaje.largoPaso(10.0), 0.45 * nominal)
        self.assertAlmostEqual(montaje.largoPaso(400.0), 1.30 * nominal)

    def test_calibrado_el_paso_es_fijo(self):
        """con las dos caminatas reales el paso del usuario salio de
        0.43-0.46 m a cualquier cadencia: ya calibrado no se escala."""
        import montaje
        montaje.LARGO_PASO_M, montaje.PASO_CALIBRADO = 0.44, True
        for cad in (None, 60.0, 83.0, 120.0):
            self.assertEqual(montaje.largoPaso(cad), 0.44)
        s = SensoresGafas(iniciar=False)
        _caminata(s, pasos=10, cadaS=0.6)
        s.tomarMovimiento()
        self.assertAlmostEqual(s.avanceM, 4.4, delta=0.01)

    def test_un_paso_perdido_no_cambia_la_cadencia(self):
        d = StepDetector(threshold=1.1, resetLevel=0.4, maxGapS=1.4)
        for hueco in (0.8, 0.8, 1.35, 0.8):       # uno casi doble
            d._anotarHueco(hueco)
        self.assertAlmostEqual(d.cadencia, 75.0, delta=0.5)
        d._anotarHueco(3.0)                        # se detuvo
        self.assertIsNone(d.cadencia)

    def test_estimador_avanza_por_cadencia(self):
        import montaje
        s = SensoresGafas(iniciar=False)
        imu = ImuGafas(s, montajePath="/no/existe.json")
        s.procesarLinea(_imu(0))
        est = MotionEstimator(imu=imu, stepLengthM=0.68, porCadencia=True)
        est.update(None, 0.5)                      # el primero no cuenta
        _caminata(s, pasos=8, cadaS=0.8, ms0=20, relojReal=True)
        motion, odo = est.update(None, 0.5)
        self.assertEqual(odo["steps_total"], 8)
        self.assertAlmostEqual(motion.delta_trans,
                               8 * montaje.largoPaso(75.0), delta=0.05)
        # con --step-length queda fijo
        s2 = SensoresGafas(iniciar=False)
        s2.procesarLinea(_imu(0))
        fijo = MotionEstimator(imu=ImuGafas(s2, montajePath="/no/existe.json"),
                               stepLengthM=0.5)
        fijo.update(None, 0.5)
        _caminata(s2, pasos=8, cadaS=0.8, ms0=20, relojReal=True)
        self.assertAlmostEqual(fijo.update(None, 0.5)[0].delta_trans, 4.0)


class TestAvancePorEco(_SinCalibracion):

    def test_pared_acercandose_mide_el_avance(self):
        from sensores import avancePorEco
        # 0.6 m/s hacia una pared que empieza a 4 m: 3 m en 5 s
        ecos = [(0.1 * k, "ID"[k % 2], 400 - 6 * k) for k in range(51)]
        r = avancePorEco(ecos)
        self.assertAlmostEqual(r["metros"], 3.0, delta=0.15)
        self.assertAlmostEqual(r["segundos"], 5.0, delta=0.3)

    def test_quieto_o_sin_eco_no_avanza(self):
        from sensores import avancePorEco
        quieto = [(0.1 * k, "ID"[k % 2], 110 + (k % 3)) for k in range(60)]
        self.assertEqual(avancePorEco(quieto)["metros"], 0.0)
        sinEco = [(0.1 * k, "ID"[k % 2], 765) for k in range(60)]
        self.assertEqual(avancePorEco(sinEco)["metros"], 0.0)

    def test_brinco_de_objetivo_corta_el_tramo(self):
        from sensores import avancePorEco
        # se acerca 1.2 m a una cosa, y de golpe ve otra mas lejos
        ecos = [(0.2 * k, "I", 200 - 12 * k) for k in range(11)]
        ecos += [(2.2 + 0.2 * k, "I", 350 - 12 * k) for k in range(11)]
        r = avancePorEco(ecos)
        self.assertAlmostEqual(r["metros"], 2.4, delta=0.2)
        self.assertEqual(len(r["tramos"]), 2)

    def test_solo_cuenta_caminando(self):
        from sensores import avancePorEco
        # alguien se acerca mientras el usuario esta parado: no es avance
        ecos = [(0.1 * k, "ID"[k % 2], 400 - 6 * k) for k in range(51)]
        self.assertEqual(avancePorEco(ecos, pasosT=[])["metros"], 0.0)
        self.assertGreater(avancePorEco(ecos, pasosT=[1.0, 2.0, 3.0, 4.0])
                           ["metros"], 2.5)

    def test_resumen_de_una_grabacion(self):
        import os, tempfile
        import montaje
        from sensores import resumenCaminata, textoResumen
        from calibrar_gafas import largoDeCaminata
        # 12 pasos a 75 pasos/min, avanzando 0.5 m/s hacia una pared
        s = SensoresGafas(iniciar=False)
        lineas = []
        s.alLinea = None
        orig = s.procesarLinea
        def grabar(linea, t=None):
            lineas.append(f"{t:.4f} {linea}")
        s.procesarLinea = grabar
        _caminata(s, pasos=12, cadaS=0.8, eco=lambda t: 600 - 50 * t)
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "serie.txt")
            with open(ruta, "w") as fh:
                fh.write("\n".join(lineas) + "\n")
            r = resumenCaminata(ruta)
        self.assertEqual(r["pasos"], 12)
        self.assertAlmostEqual(r["cadenciaPpm"], 75.0, delta=1.0)
        self.assertAlmostEqual(r["metros"], 12 * montaje.largoPaso(75.0),
                               delta=0.1)
        # el ultrasonido dice 0.5 m/s -> pasos de 0.40 m
        self.assertAlmostEqual(r["eco"]["velocidad"], 0.5, delta=0.05)
        self.assertAlmostEqual(r["eco"]["largoPasoM"], 0.40, delta=0.03)
        self.assertIn("ultrasonido", textoResumen(r))
        largo, _m = largoDeCaminata(r)
        self.assertAlmostEqual(largo, 0.40, delta=0.03)
        largo, _m = largoDeCaminata(r, metros=6.0)
        self.assertAlmostEqual(largo, 0.5)
        # sin pared enfrente y sin metros no se inventa nada
        r["eco"] = None
        self.assertIsNone(largoDeCaminata(r)[0])

    def test_aprende_el_paso_al_terminar_la_corrida(self):
        import montaje
        s = SensoresGafas(iniciar=False)
        # 0.5 m/s hacia una pared, un paso cada 0.8 s -> pasos de 0.40 m
        _caminata(s, pasos=14, cadaS=0.8, eco=lambda t: 700 - 50 * t)
        e = s.pasoMedido()
        self.assertAlmostEqual(e["largoPasoM"], 0.40, delta=0.04)
        nuevo, medido = s.aprenderPaso(guardar=False)
        self.assertAlmostEqual(nuevo, medido)          # primera vez: lo medido
        # ya calibrado solo se mueve tantito hacia lo medido
        montaje.LARGO_PASO_M, montaje.PASO_CALIBRADO = 0.60, True
        nuevo, medido = s.aprenderPaso(guardar=False)
        self.assertAlmostEqual(nuevo, 0.7 * 0.60 + 0.3 * medido, delta=0.002)
        # sin pared que se acerque no se toca nada
        quieto = SensoresGafas(iniciar=False)
        _caminata(quieto, pasos=14, cadaS=0.8, eco=lambda t: 300)
        self.assertIsNone(quieto.aprenderPaso(guardar=False))


# ---------------------------------------------------------------------
# la voz: alertas que se pisan, ceder la palabra, no creerse su eco
# ---------------------------------------------------------------------
class _MotorFalso:
    """un tts de mentiras: 'habla' durante duraS o hasta que lo paran."""

    def __init__(self, duraS=0.25):
        self.duraS = duraS
        self.dichas = []
        self.cortadas = []
        import threading
        self._parar = threading.Event()

    def say(self, texto):
        self._parar.clear()
        if self._parar.wait(self.duraS):
            self.cortadas.append(texto)
        else:
            self.dichas.append(texto)

    def parar(self):
        self._parar.set()


def _colaVoz(duraS=0.25):
    import threading
    from interaccion import ColaVoz
    voz = ColaVoz(hablar=False, echo=False)
    voz.engine = _MotorFalso(duraS)
    voz._hilo = threading.Thread(target=voz._worker, daemon=True)
    voz._hilo.start()
    return voz


class TestColaVozAlertas(unittest.TestCase):

    def test_de_una_rafaga_solo_la_que_sonaba_y_la_ultima(self):
        voz = _colaVoz()
        for k in range(1, 6):
            voz.alerta(f"alerta {k}")
            time.sleep(0.03)
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.engine.dichas, ["alerta 1", "alerta 5"])
        self.assertEqual(voz.engine.cortadas, [])     # no se cortan entre si

    def test_alerta_vieja_ya_no_se_dice(self):
        voz = _colaVoz(duraS=0.4)
        voz.caducaAlertaS = 0.2
        voz.alerta("primera")
        time.sleep(0.05)
        voz.alerta("segunda")           # espera 0.35 s: ya caduco
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.engine.dichas, ["primera"])

    def test_alerta_corta_la_platica_pero_no_al_reves(self):
        voz = _colaVoz(duraS=0.4)
        voz.decir("una respuesta larga", prioridad=2)
        time.sleep(0.1)
        voz.alerta("cuidado")
        time.sleep(0.1)
        voz._interrumpir()              # un pip critico: la alerta sigue
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.engine.cortadas, ["una respuesta larga"])
        self.assertEqual(voz.engine.dichas, ["cuidado"])

    def test_callar_tira_lo_pendiente_menos_las_alertas(self):
        voz = _colaVoz(duraS=0.3)
        voz.decir("frase uno", prioridad=2)
        voz.decir("frase dos", prioridad=2)
        time.sleep(0.1)
        voz.callar()                    # el usuario pico el boton
        voz.decir("respuesta al usuario", prioridad=2)
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.engine.cortadas, ["frase uno"])
        self.assertEqual(voz.engine.dichas, ["respuesta al usuario"])

    def test_la_voz_normal_espera_al_usuario_la_alerta_no(self):
        voz = _colaVoz(duraS=0.05)
        hablando = [True]
        voz.esperarUsuarioSi = lambda: hablando[0]
        voz.decir("instruccion", prioridad=1)
        voz.alerta("cuidado")
        time.sleep(0.3)
        self.assertEqual(voz.engine.dichas, ["cuidado"])
        hablando[0] = False
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.engine.dichas, ["cuidado", "instruccion"])

    def test_registro_de_lo_dicho(self):
        voz = _colaVoz(duraS=0.2)
        t0 = time.time()
        voz.decir("gira a la derecha", prioridad=1)
        time.sleep(0.1)
        self.assertEqual(voz.dichoEntre(time.time() - 0.05, time.time()),
                         ["gira a la derecha"])
        voz.cerrar(esperaS=5)
        self.assertEqual(voz.dichoEntre(t0 - 5.0, t0 - 1.0), [])
        self.assertEqual(voz.dichoEntre(time.time() + 2.0, time.time() + 3.0),
                         [])


class TestOidoEco(unittest.TestCase):

    def _oido(self):
        return TestOidoBoton()._oido("continuo")

    def _r(self, *palabras):
        """resultado de vosk: (palabra, inicio, fin[, conf])"""
        return {"text": " ".join(p[0] for p in palabras),
                "result": [{"word": p[0], "start": p[1], "end": p[2],
                            "conf": p[3] if len(p) > 3 else 1.0}
                           for p in palabras]}

    def test_palabras_que_se_van_a_oir(self):
        from interaccion import _palabrasEco, _pareceEco
        dichas = _palabrasEco("¡Cuidado! Avanza 3 metros, a 45 centímetros.")
        for w in ("cuidado", "avanza", "tres", "metros", "cuarenta", "cinco",
                  "centimetros"):
            self.assertIn(w, dichas)
        self.assertTrue(_pareceEco("centímetro", dichas))    # oido a medias
        self.assertFalse(_pareceEco("pausa", dichas))

    def test_sin_eco_el_usuario_puede_hablar_encima(self):
        """con la diadema la voz no se cuela (medido): lo que se oye
        mientras el sistema habla es el usuario, aunque coincida una
        palabra ("que hay enfrente" sobre "al frente...")."""
        from interaccion import _palabrasEco
        oido = self._oido()
        oido._tramosEco = [(10.0, 18.0, _palabrasEco(
            "Al frente, una silla a dos metros."))]
        r = self._r(("qué", 12.0, 12.2), ("hay", 12.2, 12.4),
                    ("enfrente", 12.4, 12.9))
        self.assertEqual(oido._sinEco(r), "qué hay enfrente")
        self.assertEqual(oido._ecoOido, 0)

    def test_su_propia_voz_no_cuenta(self):
        from interaccion import _palabrasEco
        oido = self._oido()
        oido._tramosEco = [(10.0, 14.0, _palabrasEco(
            "¡Cuidado, una pared enfrente! Detente."))]
        # el micro oyo al sistema (con una palabra mal oida) y luego,
        # ya en silencio, al usuario
        r = self._r(("cuidado", 10.2, 10.6), ("una", 10.6, 10.8),
                    ("pared", 10.8, 11.2), ("de", 11.2, 11.3),
                    ("enfrente", 11.3, 11.9), ("detente", 12.2, 12.8),
                    ("pausa", 15.0, 15.5))
        self.assertEqual(oido._sinEco(r), "pausa")
        self.assertEqual(oido._ecoOido, 5)
        # solo eco: no se entrega nada
        self.assertEqual(oido._sinEco(self._r(("detente", 12.2, 12.8))), "")

    def test_fuera_de_la_voz_del_sistema_todo_vale(self):
        from interaccion import _palabrasEco
        oido = self._oido()
        oido._tramosEco = [(10.0, 12.0, _palabrasEco("¿A dónde quieres ir?"))]
        r = self._r(("a", 13.0, 13.1), ("la", 13.1, 13.2),
                    ("cocina", 13.2, 13.8))
        self.assertEqual(oido._sinEco(r), "a la cocina")

    def test_si_el_micro_oye_la_bocina_pasa_a_medio_duplex(self):
        from interaccion import _palabrasEco
        oido = self._oido()
        dichas = _palabrasEco("Gira a la derecha y avanza tres metros.")
        oido._tramosEco = [(0.0, 5.0, dichas)]
        r = self._r(("gira", 0.2, 0.5), ("a", 0.5, 0.6), ("la", 0.6, 0.7),
                    ("derecha", 0.7, 1.2), ("y", 1.2, 1.3),
                    ("avanza", 1.3, 1.8), ("tres", 1.8, 2.1),
                    ("metros", 2.1, 2.6))
        self.assertEqual(oido._sinEco(r), "")
        self.assertTrue(oido.medioDuplex)
        # ya en medio duplex: lo que se graba mientras habla se tira
        oido.ecoFn = lambda t0, t1: ["Gira a la derecha"]
        for _ in range(6):
            oido.alimentar(b"\0" * 4000)
        self.assertTrue(oido._cola.empty())
        oido.ecoFn = lambda t0, t1: []
        for _ in range(3):
            oido.alimentar(b"\0" * 4000)
        self.assertEqual(oido._cola.get_nowait(), "pausa")

    def test_avisa_en_cuanto_entiende_algo(self):
        oido = self._oido()
        oidas = []
        oido.alOir = oidas.append
        for _ in range(3):
            oido.alimentar(b"\0" * 4000)
        self.assertEqual(oidas, ["pausa"])

    def test_recien_abierto_con_el_boton_se_le_espera(self):
        oido = TestOidoBoton()._oido("boton")
        self.assertFalse(oido.usuarioHablando())
        oido.alternar()
        self.assertTrue(oido.usuarioHablando())     # acaba de picar
        oido._tAbierto = time.time() - 5.0
        self.assertFalse(oido.usuarioHablando())    # no dijo nada
        oido._tParcial = time.time()
        self.assertTrue(oido.usuarioHablando())     # va a media frase


class TestCamaraQueFalla(unittest.TestCase):
    """2026-11-10: con el cable plano flojo la camara se detecta pero no
    da imagen; el main caia a cv2.VideoCapture y corria a ciegas."""

    def _conCamara(self, ok, detectada):
        import camara

        class Falsa:
            def __init__(self, **k):
                self.ok, self.detectada = ok, detectada
                self.razon = "" if ok else "no entrega imagen"

            def isOpened(self):
                return self.ok
        real = camara.CamaraPi
        camara.CamaraPi = Falsa
        try:
            return camara.abrirCamara(0)
        finally:
            camara.CamaraPi = real

    def test_detectada_pero_sin_imagen_no_cae_a_opencv(self):
        cap, perfil = self._conCamara(ok=False, detectada=True)
        self.assertFalse(cap.isOpened())
        self.assertIn("imagen", cap.razon)       # el main lo dice y se sale
        self.assertEqual(perfil, "picam3wide")

    def test_camara_buena(self):
        cap, perfil = self._conCamara(ok=True, detectada=True)
        self.assertTrue(cap.isOpened())
        self.assertEqual(perfil, "picam3wide")

    def test_sin_imagen_a_media_corrida(self):
        import camara
        cam = camara.CamaraPi.__new__(camara.CamaraPi)
        cam.ok, cam.tFrame = True, time.time()
        self.assertFalse(cam.sinImagen(5.0))
        cam.tFrame = time.time() - 9.0
        self.assertTrue(cam.sinImagen(5.0))
