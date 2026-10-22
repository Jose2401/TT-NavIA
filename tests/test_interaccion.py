"""
pruebas de la capa de interaccion (VozYPLN + navigator + tts) sin
hardware: pln real (es puro python), tts con motor falso, frames falsos.
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
        self.assertEqual(msgs[0].text, "Ruta calculada. Iniciando navegación.")
        self.assertEqual(self.inter.navigator.status.session, "activo")
        self.assertEqual(self.inter.navigator.status.goal_label, "door")

    def test_sesion_completa_frases_naturales(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("quiero ir a la puerta", frame)
        self.assertEqual(self.inter.manejarTexto("pausa por favor")[0].text,
                         "Navegación pausada.")
        self.assertEqual(self.inter.manejarTexto("sigue con la ruta")[0].text,
                         "Reanudando navegación.")
        self.assertEqual(self.inter.manejarTexto("ya quiero parar")[0].text,
                         "Navegación detenida.")
        self.assertEqual(self.inter.navigator.status.session, "en_espera")

    def test_aclaracion_de_destino(self):
        frame, _ = _frame(landmarks=PUERTA)
        msgs = self.inter.manejarTexto("quiero ir", frame)
        self.assertTrue(self.inter.pendienteDestino)
        self.assertIn("lugar", msgs[0].text.lower())
        msgs = self.inter.manejarTexto("la salida", frame)
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

    def test_ruta_alternativa(self):
        frame, _ = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        msgs = self.inter.manejarTexto("busca otra ruta", frame)
        self.assertEqual(msgs[0].text, "Calculando ruta alternativa.")

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
        self.assertIn("derecha", texto.lower())

    def test_repetir_ultima_instruccion(self):
        frame, mgr = _frame(landmarks=PUERTA)
        self.inter.manejarTexto("ve a la salida", frame)
        self.inter.ciclo(frame, mgr)
        msgs = self.inter.manejarTexto("repite", frame)
        self.assertEqual(msgs[0].text, self.inter.ultimaInstruccion)

    def test_desconocido_pide_repeticion(self):
        msgs = self.inter.manejarTexto("xyzzy foo bar")
        self.assertEqual(msgs[0].priority, 2)
        self.assertTrue(msgs[0].text)


class TestColaVoz(unittest.TestCase):

    def test_sin_tts_solo_imprime(self):
        voz = ColaVoz(hablar=False, echo=False)
        voz.decir("hola", prioridad=1)   # no debe tronar
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


@unittest.skipUnless(_pickModel(), "falta el modelo entrenado")
class TestArmado(unittest.TestCase):

    def test_fabrica_con_fallbacks(self):
        nav = Navigator(_pickModel())
        inter, voz, oido = armarInteraccion(nav, usarVoz=True, usarTts=False,
                                            voskModel="/no/existe")
        self.assertIsNotNone(inter)
        self.assertIsNone(oido)      # sin vosk: teclado y ya
        msgs = inter.manejarTexto("ayuda")
        self.assertTrue(msgs)


if __name__ == "__main__":
    unittest.main()
