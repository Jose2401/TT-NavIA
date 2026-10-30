"""
interaccion.py — une VozYPLN (pln + dialogo + Tts + vosk) con el
navegador: mic -> NavigationNLP -> NavCommand/acciones -> Speaker con
cola de prioridades (0 alerta interrumpe, 1 navegacion, 2 respuesta).
el Navigator habla con las frases literales del documento; aqui dialogo
las "traduce" al tono amigable del equipo. todo con fallbacks: sin mic
o sin tts el sistema sigue por teclado/consola.
"""
import math
import queue
import threading
import time

from navigation.navigator import NavCommand, NavMessage


def _vozypln():
    import sys, os
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "VozYPLN")
    if ruta not in sys.path:
        sys.path.insert(0, ruta)


class ColaVoz:
    # cola con prioridades sobre el Speaker del equipo (edge-tts ->
    # pyttsx3 -> consola, lo que haya)
    def __init__(self, hablar=True, rate=165, echo=True):
        self.echo = echo
        self.engine = None
        self.hablando = False
        self._cola = queue.PriorityQueue()
        self._n = 0
        self._ultima = {}
        self._canalN = {}
        self.caducaS = 6.0      # una frase de canal mas vieja que esto ya no se dice
        if hablar:
            try:
                _vozypln()
                from Tts import Speaker
                self.engine = Speaker()
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception as e:
                print(f"[VOZ] tts no disponible ({e}), solo consola")

    def decir(self, texto, prioridad=2, cadaS=0.0, canal=None):
        # cadaS > 0: no repetir la misma frase antes de ese tiempo.
        # canal: las frases de un canal se pisan entre si — solo se habla
        # la MAS RECIENTE (una indicacion vieja de "gira" ya no sirve y
        # era lo que desfasaba la voz cuando el tts no alcanzaba)
        if cadaS > 0:
            ahora = time.time()
            if ahora - self._ultima.get(texto, 0.0) < cadaS:
                return
            self._ultima[texto] = ahora
        if self.echo:
            print(f"[VOZ p{prioridad}] {texto}", flush=True)
        if self.engine is None:
            return
        if prioridad == 0:
            self._interrumpir()
        self._n += 1
        if canal:
            self._canalN[canal] = self._n
        self._cola.put((prioridad, self._n, texto, canal, time.time()))

    def decirMsgs(self, msgs):
        for m in msgs:
            canal = "instruccion" if m.kind == "instruccion" else None
            self.decir(m.text, m.priority, canal=canal)

    def _interrumpir(self):
        # p0 corta lo que se este reproduciendo (edge usa pygame)
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if getattr(self.engine, "_fallback", None):
                self.engine._fallback.stop()
        except Exception:
            pass

    def _worker(self):
        while True:
            _p, n, texto, canal, t0 = self._cola.get()
            if canal:
                # si ya llego una mas nueva del mismo canal, esta se tira
                if n < self._canalN.get(canal, 0):
                    continue
                if time.time() - t0 > self.caducaS:
                    continue
            self.hablando = True
            try:
                self.engine.say(texto)    # Speaker.say bloquea hasta acabar
            except Exception:
                pass
            self.hablando = False


class OidoVoz:
    # mic -> texto con vosk; corre en su hilo y mete frases a una cola
    def __init__(self, modelPath, device=None):
        self.ok = False
        self.razon = ""
        self._rec = None
        try:
            import sys, os
            ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "VozYPLN")
            if ruta not in sys.path:
                sys.path.insert(0, ruta)
            from voz_recog import VoiceRecognizer
            self._rec = VoiceRecognizer(modelPath, device=device)
            self.ok = True
        except FileNotFoundError as e:
            self.razon = str(e)
        except ImportError as e:
            self.razon = f"falta dependencia: {e}"
        except Exception as e:
            self.razon = str(e)

    def start(self, colaTexto, muteFn=lambda: False):
        if not self.ok:
            return
        def loop():
            while True:
                try:
                    texto = self._rec.listen()
                except Exception as e:
                    print(f"[VOZ] mic fallo: {e}")
                    time.sleep(2)
                    continue
                # si el tts estaba hablando, lo mas seguro es que el mic
                # se oyo a si mismo
                if texto and not muteFn():
                    colaTexto.put(texto)
        threading.Thread(target=loop, daemon=True).start()


AYUDA_VOZ = ("Puede decir: llévame a un lugar, pausa, continúa, detener, "
             "otra ruta, dónde estoy, qué hay enfrente, o repite.")

_ARTICULOS = ("el ", "la ", "los ", "las ", "un ", "una ", "al ")


def _limpiarDestino(destino):
    # el pln a veces deja el articulo ("la puerta") o plural ("banos");
    # el navigator espera la palabra pelona pa sus alias
    if not destino:
        return destino
    d = destino.strip().lower()
    for art in _ARTICULOS:
        if d.startswith(art):
            d = d[len(art):]
            break
    from navigation.navigator import DEST_ALIASES
    if d not in DEST_ALIASES and d.endswith("s") and d[:-1] in DEST_ALIASES:
        d = d[:-1]
    return d


# confirmaciones literales del navigator que dialogo sabe decir bonito
_LITERALES_AMIGABLES = {"Ruta calculada. Iniciando navegación.",
                        "Navegación pausada.", "Reanudando navegación.",
                        "Navegación detenida.",
                        "Calculando ruta alternativa."}
_SALIR = {"apagar sistema", "terminar programa"}


class Interaccion:
    """recibe texto (de voz o teclado), lo pasa por el pln del equipo y
    ejecuta: lo de sesion va al Navigator, lo demas (entorno, repetir,
    emergencia, ayuda...) se resuelve aqui. con amigable=True las frases
    al usuario salen con el tono de dialogo.py (la "Guía" del equipo);
    el Navigator por dentro sigue con las literales del documento."""

    def __init__(self, navigator, nlp=None, amigable=True):
        _vozypln()
        if nlp is None:
            from PLN_oneimport import NavigationNLP
            nlp = NavigationNLP()
        import dialogo
        self.dialogo = dialogo
        self.nlp = nlp
        self.navigator = navigator
        self.amigable = amigable
        self.pendienteDestino = False
        self.ultimaInstruccion = None

    def _bonito(self, r):
        return self.dialogo.friendly_response(r) if self.amigable else None

    def manejarTexto(self, texto, navFrame=None):
        if not texto or not texto.strip():
            return []

        from PLN_oneimport import normalize_text
        if normalize_text(texto) in _SALIR:
            return [NavMessage(self.dialogo.goodbye(), priority=2,
                               kind="despedida")]

        # atajo de desarrollo: "punto 5 3" va directo, sin pasar por pln
        from navigation.navigator import parse_command
        rapido = parse_command(texto)
        if rapido.tipo == "navegar" and rapido.punto is not None:
            return self.navigator.handle_command(rapido, navFrame)

        # veniamos de "¿a que lugar quieres ir?": lo que diga es el destino
        if self.pendienteDestino:
            self.pendienteDestino = False
            r = self.nlp.process(texto)
            if r["accion"] not in ("denegar", "detener_navegacion",
                                   "cancelar_ruta"):
                texto = f"llevame a {texto}"

        r = self.nlp.process(texto)
        accion = r["accion"]

        if accion == "navegar":
            if r["requiere_aclaracion"]:
                self.pendienteDestino = True
                pregunta = (self.dialogo.ask_destination() if self.amigable
                            else r["respuesta_sugerida"])
                return [NavMessage(pregunta, priority=2, kind="respuesta")]
            cmd = NavCommand("navegar", destino=_limpiarDestino(r["destino"]),
                             texto_original=texto)
            return self._alNavigator(cmd, navFrame, r)

        simple = {"pausar_navegacion": "pausar",
                  "reanudar_navegacion": "reanudar",
                  "detener_navegacion": "detener",
                  "cancelar_ruta": "detener",
                  "recalcular_ruta": "ruta_alternativa",
                  "consultar_ubicacion": "estado",
                  "consultar_estado_ruta": "estado"}
        if accion in simple:
            cmd = NavCommand(simple[accion], texto_original=texto)
            return self._alNavigator(cmd, navFrame, r)

        if accion == "describir_entorno":
            return [NavMessage(self.describirEntorno(navFrame),
                               priority=1, kind="respuesta")]
        if accion == "repetir_ultima_instruccion":
            if self.ultimaInstruccion:
                return [NavMessage(self.ultimaInstruccion, priority=1,
                                   kind="instruccion")]
            return [NavMessage("No hay ninguna instrucción reciente.",
                               priority=2, kind="respuesta")]
        if accion == "notificar_emergencia":
            aviso = self._bonito(r) or "Emergencia activada. Deteniéndome."
            msgs = [NavMessage(aviso, priority=0, kind="alerta")]
            if self.navigator.status.session == "activo":
                msgs += self.navigator.handle_command(
                    NavCommand("pausar"), navFrame)
            # TODO: aqui va el aviso real a contactos cuando ese modulo exista
            return msgs
        if accion == "mostrar_ayuda":
            return [NavMessage(self._bonito(r) or AYUDA_VOZ, priority=2,
                               kind="respuesta")]
        if accion == "confirmar":
            return [NavMessage(self._bonito(r) or "De acuerdo.",
                               priority=2, kind="respuesta")]
        if accion == "denegar":
            self.pendienteDestino = False
            return [NavMessage(self._bonito(r) or "Entendido, cancelado.",
                               priority=2, kind="respuesta")]

        # DESCONOCIDO
        return [NavMessage(self._bonito(r) or r["respuesta_sugerida"],
                           priority=2, kind="respuesta")]

    def _alNavigator(self, cmd, navFrame, r):
        # las confirmaciones exitosas se dicen con el tono de dialogo;
        # los errores ("No conozco...") se quedan tal cual, son informativos
        msgs = self.navigator.handle_command(cmd, navFrame)
        if self.amigable:
            for m in msgs:
                if m.text in _LITERALES_AMIGABLES:
                    m.text = self.dialogo.friendly_response(r)
        return msgs

    def ciclo(self, navFrame, mgr, alertas=None):
        label = self.navigator.status.goal_label
        msgs = self.navigator.update(navFrame, mgr, alertas)
        for m in msgs:
            if self.amigable:
                if m.kind == "instruccion" and m.action is not None:
                    amigo = self._instruccionAmigable(m.action, navFrame)
                    if amigo:
                        m.text = amigo
                elif m.text == "Ha llegado a su destino.":
                    m.text = self.dialogo.arrival_message(label)
            if m.kind == "instruccion":
                self.ultimaInstruccion = m.text
        return msgs

    def _instruccionAmigable(self, action, navFrame):
        # ojo signos: dialogo usa grados>0 = derecha; nuestro phi>0 = izq
        st = self.navigator.status
        if action == 3 or st.phi_goal is None:
            return None                       # "Deténgase" se queda literal
        grados = max(20, round(abs(math.degrees(st.phi_goal))))
        if action == 0:
            dist = st.d_goal
            if navFrame is not None and dist is not None:
                dist = min(dist, max(0.5, navFrame.d_front))
            return self.dialogo.navigation_instruction(0, dist)
        if action == 1:
            return self.dialogo.navigation_instruction(-grados)
        return self.dialogo.navigation_instruction(grados)

    def describirEntorno(self, frame):
        if frame is None:
            return "Aún no tengo información del entorno."
        partes = []
        for nombre, lo, hi in (("Al frente", -math.radians(30),
                                math.radians(30)),
                               ("A la izquierda", math.radians(30),
                                math.radians(90)),
                               ("A la derecha", -math.radians(90),
                                -math.radians(30))):
            cerca = None
            for d in frame.detections:
                if lo <= d.bearing <= hi and d.range_est is not None:
                    if cerca is None or d.range_est < cerca.range_est:
                        cerca = d
            if cerca is not None:
                extra = " en movimiento" if cerca.moving else ""
                tag = "peligro de piso" if cerca.hazard else cerca.label
                partes.append(f"{nombre}, {tag}{extra} a "
                              f"{cerca.range_est:.0f} metros"
                              if cerca.range_est >= 1.5 else
                              f"{nombre}, {tag}{extra} a "
                              f"{int(cerca.range_est * 100)} centímetros")
            else:
                d_libre = {"Al frente": frame.d_front,
                           "A la izquierda": frame.d_left,
                           "A la derecha": frame.d_right}[nombre]
                if d_libre >= 3.99:
                    partes.append(f"{nombre} libre")
                else:
                    partes.append(f"{nombre}, obstáculo a "
                                  f"{d_libre:.0f} metros"
                                  if d_libre >= 1.5 else
                                  f"{nombre}, obstáculo a "
                                  f"{int(d_libre * 100)} centímetros")
        if frame.room:
            partes.append(f"Está en {frame.room}")
        return ". ".join(partes) + "."


def armarInteraccion(navigator, usarVoz=False, usarTts=False,
                     voskModel=None, micDevice=None, echo=True):
    """fabrica todo el stack de interaccion con fallbacks: sin modelo
    vosk o sin mic avisa y sigue solo con teclado."""
    voz = ColaVoz(hablar=usarTts, echo=echo)
    inter = Interaccion(navigator)
    oido = None
    if usarVoz:
        oido = OidoVoz(voskModel or "models/vosk-model-small-es-0.42",
                       device=micDevice)
        if oido.ok:
            print("[VOZ] micrófono + vosk listos")
        else:
            print(f"[VOZ] reconocimiento no disponible: {oido.razon}")
            print("[VOZ] seguimos con comandos por teclado")
            oido = None
    return inter, voz, oido
