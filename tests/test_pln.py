"""
evaluacion del pln del equipo (rnf-07 pide >= 92 %). FRASES incluye las
que fallaban en la evaluacion del documento; APARTADO son frases que NO
estan en la semilla (mide generalizacion, no memoria). corre con una
memoria limpia en un dir temporal pa no tocar la del repo.
correr: python -m unittest tests.test_pln
"""
import os
import sys
import tempfile
import unittest

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "VozYPLN"))
sys.path.insert(0, os.path.join(_RAIZ, "tests"))

from frases_pln import FRASES, CRITICAS, APARTADO

GRAVES = {("DETENER", "REANUDAR"), ("REANUDAR", "DETENER"),
          ("NEGAR", "CONFIRMAR"), ("CONFIRMAR", "NEGAR"),
          ("EMERGENCIA", "AYUDA"), ("DETENER", "DESTINO"),
          ("DETENER", "PAUSA")}


class TestPLN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp(prefix="pln_"))
        from PLN_oneimport import NavigationNLP
        cls.nlp = NavigationNLP()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _intencion(self, t):
        return self.nlp.process(t)["intencion"]

    def test_criticas_todas_bien(self):
        for t, esp in FRASES:
            if t in CRITICAS:
                self.assertEqual(self._intencion(t), esp, t)

    def test_set_completo_supera_rnf07(self):
        ok = sum(self._intencion(t) == e for t, e in FRASES)
        self.assertGreaterEqual(ok / len(FRASES), 0.92)

    def test_apartado_generaliza_sin_invertir_ordenes(self):
        ok, graves = 0, []
        for t, esp in APARTADO:
            r = self._intencion(t)
            ok += r == esp
            if (esp, r) in GRAVES:
                graves.append((t, r))
        self.assertEqual(graves, [])
        self.assertGreaterEqual(ok / len(APARTADO), 0.85)

    def test_lesion_gana_aunque_diga_ayuda_o_cancela(self):
        self.assertEqual(self._intencion("ayuda me caí, cancela todo"),
                         "EMERGENCIA")

    def test_firma_cambia_si_cambian_las_frases(self):
        import PLN_oneimport as p
        antes = p._firma_semilla()
        p.INTENTS["AYUDA"].append("frase de prueba xyz")
        try:
            self.assertNotEqual(antes, p._firma_semilla())
        finally:
            p.INTENTS["AYUDA"].pop()
        self.assertEqual(antes, p._firma_semilla())


if __name__ == "__main__":
    unittest.main()
