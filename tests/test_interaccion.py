"""
pruebas de la capa de interaccion (VozYPLN: pln + dialogo + tts) sin
hardware. las frases de dialogo.py son aleatorias, asi que se asierta
por estado de sesion y contenido, no por texto exacto.
correr: python -m unittest tests.test_interaccion
"""
import math
import os
import unittest

from interaccion import ColaVoz, OidoVoz, Interaccion, armarInteraccion
from navigation.navigator import Navigator


def _pickModel():
    for base in ("models/nav_ppo", "models/nav_ppo_extended_cmp"):
        if os.path.exists(base + "_policy.npz"):
            return base
    return None


def _frame(x=0.0, y=0.0, landmarks=None, detections=None):
    from ekf_slam import EKFSlam
    from map_manager import ChunkedMapManager
    from navigation_feed import buildNavigationFrame
    slam = EKFSlam(x, y, 0.0)
    if landmarks:
        slam.importLandmarks(landmarks)
    mgr = ChunkedMapManager(mapDir=None, chunkSizeM=4.0)
    return buildNavigationFrame(slam, mgr, detections or []), mgr


PUERTA = [{"x": 3.0, "y": 0.0, "label": "door", "seen_count": 5,
           "required_hits": 2}]


@unittest.skipUnless(_pickModel(), "falta el modelo entrenado")
class TestInteraccionFlujos(unittest.TestCase):

    def setUp(self):
        nav = Navigator(_pickModel(), repeat_s=0.0, decide_every_s=0.0)
        self.inter = Interaccion(nav)

    def test_destino_por_voz(self):
        frame, _ = _frame(landmarks=PUERTA)
        msgs = self.inter.manejarTexto("llevame a la salida", frame)
        self.assertTrue(msgs and msgs[0].text)
        self.assertEqual(self.inter.navigator.status.session, "activo")
        self.assertEqual(self.inter.navigator.status.goal_label, "door")

    def test_sesion_completa_frases_naturales(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("quiero ir a la puerta", frame)
        self.assertEqual(self.inter.navigator.status.session, "activo")
        self.inter.manejarTexto("pausa por favor")
        self.assertEqual(self.inter.navigator.status.session, "pausado")
        self.inter.manejarTexto("sigue con la ruta")
        self.assertEqual(self.inter.navigator.status.session, "activo")
        self.inter.manejarTexto("ya quiero parar")
        self.assertEqual(self.inter.navigator.status.session, "en_espera")

    def test_sin_amigable_quedan_las_frases_del_documento(self):
        nav = Navigator(_pickModel(), repeat_s=0.0, decide_every_s=0.0)
        inter = Interaccion(nav, amigable=False)
        frame, _ = _frame(landmarks=PUERTA)
        msgs = inter.manejarTexto("llevame a la salida", frame)
        self.assertEqual(msgs[0].text, "Ruta calculada. Iniciando navegación.")
        self.assertEqual(inter.manejarTexto("pausa")[0].text,
                         "Navegación pausada.")

    def test_aclaracion_de_destino(self):
        frame, _ = _frame(landmarks=PUERTA)
        msgs = self.inter.manejarTexto("quiero ir", frame)
        self.assertTrue(self.inter.pendienteDestino)
        self.assertIn("?", msgs[0].text)
        self.inter.manejarTexto("la salida", frame)
        self.assertFalse(self.inter.pendienteDestino)
        self.assertEqual(self.inter.navigator.status.session, "activo")

    def test_aclaracion_cancelada_con_no(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("quiero ir", frame)
        self.inter.manejarTexto("no", frame)
        self.assertFalse(self.inter.pendienteDestino)
        self.assertEqual(self.inter.navigator.status.session, "en_espera")

    def test_punto_directo_sin_pln(self):
        frame, _ = _frame()
        self.inter.manejarTexto("punto 2 1", frame)
        self.assertEqual(self.inter.navigator.status.goal, (2.0, 1.0))

    def test_emergencia_es_p0_y_pausa(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        msgs = self.inter.manejarTexto("auxilio", frame)
        self.assertEqual(msgs[0].priority, 0)
        self.assertEqual(self.inter.navigator.status.session, "pausado")

    def test_ruta_alternativa_mantiene_sesion(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        msgs = self.inter.manejarTexto("busca otra ruta", frame)
        self.assertTrue(msgs and msgs[0].text)
        self.assertEqual(self.inter.navigator.status.session, "activo")

    def test_describir_entorno(self):
        from interfaces import VisionDetection
        dets = [VisionDetection(bearing=0.0, label="chair", range_est=1.8),
                VisionDetection(bearing=math.radians(60), label="hoyo",
                                range_est=1.0, hazard=True)]
        frame, _ = _frame(detections=dets)
        msgs = self.inter.manejarTexto("que hay enfrente", frame)
        texto = msgs[0].text
        self.assertIn("chair", texto)
        self.assertIn("peligro de piso", texto)

    def test_repetir_ultima_instruccion(self):
        frame, mgr = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        self.inter.ciclo(frame, mgr)
        msgs = self.inter.manejarTexto("repite", frame)
        self.assertEqual(msgs[0].text, self.inter.ultimaInstruccion)

    def test_instrucciones_amigables_y_signos(self):
        # phi>0 = goal a la IZQUIERDA; dialogo debe decir izquierda
        frame, mgr = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        st = self.inter.navigator.status
        st.phi_goal = math.radians(60)
        txt_izq = self.inter._instruccionAmigable(1, frame)
        self.assertIn("izquierda", txt_izq)
        txt_der = self.inter._instruccionAmigable(2, frame)
        self.assertIn("derecha", txt_der)
        txt_avanza = self.inter._instruccionAmigable(0, frame)
        self.assertTrue("paso" in txt_avanza or "derech" in txt_avanza
                        or "frente" in txt_avanza)

    def test_llegada_con_dialogo(self):
        frame, mgr = _frame(x=2.8, y=0.0, landmarks=PUERTA)
        self.inter.manejarTexto("ve a la puerta", frame)
        msgs = self.inter.ciclo(frame, mgr)
        self.assertTrue(msgs)
        t = msgs[0].text.lower()
        self.assertTrue("llega" in t or "lograste" in t, t)
        self.assertEqual(self.inter.navigator.status.session, "en_espera")

    def test_apagar_sistema(self):
        msgs = self.inter.manejarTexto("apagar sistema")
        self.assertEqual(msgs[0].kind, "despedida")

    def test_desconocido_pide_repeticion(self):
        msgs = self.inter.manejarTexto("xyzzy foo bar")
        self.assertEqual(msgs[0].priority, 2)
        self.assertTrue(msgs[0].text)


class TestColaVoz(unittest.TestCase):

    def test_sin_tts_solo_imprime(self):
        voz = ColaVoz(hablar=False, echo=False)
        voz.decir("hola", prioridad=1)
        self.assertIsNone(voz.engine)

    def test_cadaS_no_repite(self):
        voz = ColaVoz(hablar=False, echo=False)
        import io, contextlib
        voz.echo = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            voz.decir("alerta", prioridad=0, cadaS=5.0)
            voz.decir("alerta", prioridad=0, cadaS=5.0)
            voz.decir("otra", prioridad=1)
        salida = buf.getvalue()
        self.assertEqual(salida.count("alerta"), 1)
        self.assertEqual(salida.count("otra"), 1)

    def test_prioridad_ordena_cola(self):
        voz = ColaVoz(hablar=False, echo=False)
        voz._n += 1; voz._cola.put((2, voz._n, "conf"))
        voz._n += 1; voz._cola.put((0, voz._n, "alerta"))
        voz._n += 1; voz._cola.put((1, voz._n, "instr"))
        orden = [voz._cola.get()[2] for _ in range(3)]
        self.assertEqual(orden, ["alerta", "instr", "conf"])


class TestOidoVoz(unittest.TestCase):

    def test_sin_modelo_no_truena(self):
        oido = OidoVoz("/ruta/que/no/existe")
        self.assertFalse(oido.ok)
        self.assertTrue(oido.razon)


class TestDialogo(unittest.TestCase):
    # el modulo del equipo, probado directo

    def setUp(self):
        import sys, os
        ruta = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "VozYPLN")
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
        import dialogo
        self.dialogo = dialogo

    def test_describe_turn_buckets(self):
        self.assertIsNone(self.dialogo.describe_turn(5))
        self.assertIn("derecha", self.dialogo.describe_turn(45))
        self.assertIn("izquierda", self.dialogo.describe_turn(-45))
        self.assertIn("vuelta", self.dialogo.describe_turn(170))

    def test_distancias_en_pasos(self):
        self.assertIn("pasos", self.dialogo.describe_distance(5.0))
        self.assertIn("metros", self.dialogo.describe_distance(30.0))

    def test_friendly_cubre_todas_las_intenciones(self):
        from PLN_oneimport import INTENTS
        for intent in list(INTENTS) + ["DESCONOCIDO"]:
            r = {"intencion": intent, "destino": "salida"}
            self.assertTrue(self.dialogo.friendly_response(r))


@unittest.skipUnless(_pickModel(), "falta el modelo entrenado")
class TestArmado(unittest.TestCase):

    def test_fabrica_con_fallbacks(self):
        nav = Navigator(_pickModel())
        inter, voz, oido = armarInteraccion(nav, usarVoz=True, usarTts=False,
                                            voskModel="/no/existe")
        self.assertIsNotNone(inter)
        self.assertIsNone(oido)
        msgs = inter.manejarTexto("ayuda")
        self.assertTrue(msgs)


if __name__ == "__main__":
    unittest.main()
