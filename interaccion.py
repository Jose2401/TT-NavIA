"""
interaccion.py — une VozYPLN (pln + dialogo + Tts + vosk) con el
navegador: mic -> NavigationNLP -> NavCommand/acciones -> Speaker con
cola de prioridades (0 alerta interrumpe, 1 navegacion, 2 respuesta).
el Navigator habla con las frases literales del documento; aqui dialogo
las "traduce" al tono amigable del equipo. todo con fallbacks: sin mic
o sin tts el sistema sigue por teclado/consola.
"""
import math
import os
import queue
import re
import threading
import time

from navigation.navigator import NavCommand, NavMessage


def _vozypln():
    import sys, os
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "VozYPLN")
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
    try:
        import audio_pi
        audio_pi.prepararAudio()       # antes de importar sounddevice
    except Exception:
        pass


DIR_PIPER = os.path.join("models", "piper")
VOZ_PIPER = "es_MX-claude-high"


def _modeloPiper(nombre=None):
    ruta = os.path.join(DIR_PIPER, (nombre or VOZ_PIPER) + ".onnx")
    if os.path.exists(ruta) and os.path.exists(ruta + ".json"):
        return ruta
    if nombre is None and os.path.isdir(DIR_PIPER):
        for f in sorted(os.listdir(DIR_PIPER)):      # la que haya
            if f.endswith(".onnx") and os.path.exists(
                    os.path.join(DIR_PIPER, f + ".json")):
                return os.path.join(DIR_PIPER, f)
    return None


class VozPiper:
    """voz neuronal SIN internet (piper, rnf-13): el edge-tts del equipo
    suena muy bien pero ocupa red, y caminando con los lentes no hay.
    misma interfaz que su Speaker (say bloquea hasta acabar) y tambien
    reproduce por pygame.mixer.music, asi una alerta la corta igual.
    cada frase se sintetiza una vez y se guarda en tts_cache/."""
    MAX_CACHE = 600

    def __init__(self, modelo, cacheDir="tts_cache", velocidad=1.0):
        from piper import PiperVoice
        import pygame
        self._pygame = pygame
        self.modelo = modelo
        self.voice = PiperVoice.load(modelo)
        try:
            # UN solo hilo pa sintetizar: con los 4 le quitaba los nucleos
            # a la vision (medido: 380 -> 1160 ms por cuadro mientras
            # hablaba; con 1 hilo queda en ~600 y la frase sale igual a
            # tiempo, ~0.3 s de sintesis por segundo de audio)
            import onnxruntime as ort
            op = ort.SessionOptions()
            op.intra_op_num_threads = int(os.environ.get("NAVIA_HILOS_VOZ", "1"))
            op.inter_op_num_threads = 1
            self.voice.session = ort.InferenceSession(
                modelo, sess_options=op, providers=["CPUExecutionProvider"])
        except Exception:
            pass
        self.velocidad = velocidad
        self.cacheDir = cacheDir
        self._fallback = None
        self._parar = False
        self._lock = threading.Lock()
        os.makedirs(cacheDir, exist_ok=True)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self._podar()

    def _podar(self):
        try:
            wavs = [os.path.join(self.cacheDir, f)
                    for f in os.listdir(self.cacheDir) if f.endswith(".wav")]
            if len(wavs) > self.MAX_CACHE:
                wavs.sort(key=os.path.getatime)
                for f in wavs[:len(wavs) - self.MAX_CACHE]:
                    os.remove(f)
        except OSError:
            pass

    def _ruta(self, texto):
        import hashlib
        llave = f"piper|{os.path.basename(self.modelo)}|{self.velocidad}|{texto}"
        return os.path.join(self.cacheDir,
                            hashlib.md5(llave.encode("utf-8")).hexdigest()
                            + ".wav")

    def sintetizar(self, texto):
        ruta = self._ruta(texto)
        if os.path.exists(ruta):
            return ruta
        import wave
        tmp = ruta + ".tmp"
        with self._lock:                 # el modelo no es reentrante
            cfg = None
            if self.velocidad != 1.0:
                from piper import SynthesisConfig
                cfg = SynthesisConfig(length_scale=1.0 / self.velocidad)
            with wave.open(tmp, "wb") as w:
                self.voice.synthesize_wav(texto, w, syn_config=cfg)
        os.replace(tmp, ruta)
        return ruta

    def precalentar(self, frases):
        # en otro hilo: deja sintetizadas las frases urgentes pa que la
        # primera alerta no espere a la sintesis
        def tarea():
            for f in frases:
                if self._parar:
                    return
                try:
                    self.sintetizar(f)
                except Exception:
                    return
        self._hiloCalentar = threading.Thread(target=tarea, daemon=True)
        self._hiloCalentar.start()

    def cerrar(self):
        # si el programa termina con el hilo a media sintesis, onnxruntime
        # truena feo ("terminate called..."): se le espera tantito
        self._parar = True
        hilo = getattr(self, "_hiloCalentar", None)
        if hilo is not None:
            hilo.join(timeout=5.0)

    def say(self, texto, cortar=None):
        """cortar(): si da True la frase ya no se dice (la cortaron
        mientras se sintetizaba, cuando todavia no habia nada que parar)."""
        if not texto:
            return
        ruta = self.sintetizar(texto)
        if cortar is not None and cortar():
            return
        musica = self._pygame.mixer.music
        musica.load(ruta)
        musica.play()
        while musica.get_busy():
            self._pygame.time.wait(30)
        musica.unload()


def armarMotorVoz(motor="auto"):
    """motor: "piper" (sin internet), "edge" (el Speaker del equipo:
    edge-tts con respaldo pyttsx3) o "auto" (piper si esta su modelo,
    si no el del equipo)."""
    _vozypln()
    if motor in ("auto", "piper"):
        modelo = _modeloPiper()
        if modelo is not None:
            try:
                return VozPiper(modelo)
            except Exception as e:
                print(f"[VOZ] piper no disponible ({e})")
        elif motor == "piper":
            print(f"[VOZ] falta el modelo de piper en {DIR_PIPER} "
                  "(correr instalar_pi.sh)")
    from Tts import Speaker
    return Speaker()


class ColaVoz:
    # cola con prioridades sobre el motor de voz que haya (piper sin
    # internet, o el Speaker del equipo: edge-tts -> pyttsx3 -> consola)
    ALERTA = "alerta"

    def __init__(self, hablar=True, rate=165, echo=True, motor="auto"):
        self.echo = echo
        self.engine = None
        self.hablando = False
        self._cola = queue.PriorityQueue()
        self._n = 0
        self._ultima = {}
        self._canalN = {}
        self.caducaS = 6.0      # una frase de canal mas vieja que esto ya no se dice
        # las alertas se pisan entre si y caducan rapido: de una rafaga
        # solo se oye la que ya iba sonando y LA ULTIMA; decirlas todas
        # en fila era puro ruido y la que importaba llegaba tarde
        self.caducaAlertaS = 3.0
        self.esperarSi = None   # callable: True mientras suene un pip
        # callable: True mientras el usuario va hablando (la voz normal
        # se espera; las alertas no)
        self.esperarUsuarioSi = None
        self.esperaUsuarioMaxS = 6.0
        self._actual = None     # (n, canal) de lo que se esta diciendo
        self._cortada = 0       # n de la frase que se corto
        self._corteN = 0        # callar(): lo encolado antes ya no se dice
        self._dichas = []       # [tIni, tFin|None, texto] pal filtro de eco
        if hablar:
            try:
                self.engine = armarMotorVoz(motor)
                self._hilo = threading.Thread(target=self._worker,
                                              daemon=True)
                self._hilo.start()
            except Exception as e:
                print(f"[VOZ] tts no disponible ({e}), solo consola")

    def vaciar(self):
        """espera (poquito) a que se diga lo pendiente; pa la despedida."""
        t0 = time.time()
        while time.time() - t0 < 8.0 and (self.hablando
                                          or not self._cola.empty()):
            time.sleep(0.05)

    def cerrar(self, esperaS=15.0):
        """dice lo que falte (la despedida) y APAGA el hilo de voz. si el
        programa termina con ese hilo a media sintesis, onnxruntime aborta
        feo al salir ("terminate called...", codigo 134)."""
        hilo = getattr(self, "_hilo", None)
        if hilo is not None:
            self._n += 1
            self._cola.put((99, self._n, None, None, time.time()))   # fin
            hilo.join(timeout=esperaS)
        if hasattr(self.engine, "cerrar"):
            self.engine.cerrar()

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

    def alerta(self, texto):
        """frase de una alerta de obstaculo: corta la platica normal,
        pero no a otra alerta que ya va sonando; si llegan varias
        seguidas solo se dice la ultima."""
        self.decir(texto, prioridad=0, canal=self.ALERTA)

    def decirMsgs(self, msgs):
        for m in msgs:
            canal = "instruccion" if m.kind == "instruccion" else None
            self.decir(m.text, m.priority, canal=canal)

    def callar(self):
        """el usuario pidio la palabra (pico el boton o ya dijo algo):
        se corta lo que se este diciendo y lo que estaba en fila. las
        alertas no se tocan."""
        self._corteN = self._n
        self._interrumpir()

    def _interrumpir(self, todo=False):
        # corta lo que se este reproduciendo (piper y edge usan pygame).
        # una alerta que ya va sonando NO se corta: son frases de 2 s y
        # cortarlas a medias con cada pitido era lo que las encimaba
        actual = self._actual
        if actual is None:
            return
        if actual[1] == self.ALERTA and not todo:
            return
        self._cortada = actual[0]
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if getattr(self.engine, "_fallback", None):
                self.engine._fallback.stop()
            if hasattr(self.engine, "parar"):
                self.engine.parar()
        except Exception:
            pass

    def dichoEntre(self, t0, t1, colaS=0.4):
        """textos que el sistema estaba diciendo entre t0 y t1 (reloj de
        time.time()); colaS = lo que tarda en salir el final por la
        bocina. lo usa el oido pa no creerse su propia voz."""
        out = []
        for ini, fin, texto in list(self._dichas):
            if ini <= t1 and (fin is None or fin + colaS >= t0):
                out.append(texto)
        return out

    def _vigente(self, prio, n, canal, t0):
        if canal:
            # si ya llego una mas nueva del mismo canal, esta se tira
            if n < self._canalN.get(canal, 0):
                return False
            caduca = (self.caducaAlertaS if canal == self.ALERTA
                      else self.caducaS)
            if time.time() - t0 > caduca:
                return False
        if canal != self.ALERTA and prio > 0 and n <= self._corteN:
            return False                 # callar()
        return True

    def _worker(self):
        while True:
            prio, n, texto, canal, t0 = self._cola.get()
            if texto is None:
                return                   # cerrar(): ya no hay mas que decir
            if not self._vigente(prio, n, canal, t0):
                continue
            normal = canal != self.ALERTA and prio > 0
            tEspera = time.time()
            while True:
                if self.esperarSi and self.esperarSi():
                    pass
                elif (normal and self.esperarUsuarioSi
                      and time.time() - tEspera < self.esperaUsuarioMaxS
                      and self.esperarUsuarioSi()):
                    pass
                else:
                    break
                time.sleep(0.05)
            if not self._vigente(prio, n, canal, t0):
                continue                 # se hizo vieja esperando
            self._actual = (n, canal)
            self.hablando = True
            registro = [time.time(), None, texto]
            self._dichas.append(registro)
            del self._dichas[:-30]
            try:
                if isinstance(self.engine, VozPiper):
                    self.engine.say(texto,
                                    cortar=lambda: self._cortada == n)
                else:
                    self.engine.say(texto)   # Speaker.say bloquea hasta acabar
            except Exception:
                pass
            registro[1] = time.time()
            self.hablando = False
            self._actual = None


_UNIDADES = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez "
             "once doce trece catorce quince dieciseis diecisiete dieciocho "
             "diecinueve veinte veintiuno veintidos veintitres veinticuatro "
             "veinticinco veintiseis veintisiete veintiocho veintinueve").split()
_DECENAS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta",
            70: "setenta", 80: "ochenta", 90: "noventa"}


def _sinAcentos(texto):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def _palabrasEco(texto):
    """las palabras que se van a OIR cuando el tts diga ese texto (sin
    acentos ni signos; los numeros como se pronuncian)."""
    import re
    out = set()
    for w in re.findall(r"[a-zñ]+|\d+", _sinAcentos(texto)):
        if w.isdigit():
            n = int(w)
            if n < 30:
                out.add(_UNIDADES[n])
            elif n < 100:
                out.add(_DECENAS[n - n % 10])
                if n % 10:
                    out.update(("y", _UNIDADES[n % 10]))
            else:
                out.update(("cien", "ciento", "mil"))
        else:
            out.add(w)
    return out


def _pareceEco(palabra, dichas):
    """la palabra que oyo vosk ¿es una de las que decia el sistema? se
    acepta parecida, pq el eco llega bajito y vosk lo oye a medias."""
    w = _sinAcentos(palabra)
    if w in dichas:
        return True
    if len(w) < 4:
        return False
    import difflib
    return bool(difflib.get_close_matches(w, dichas, n=1, cutoff=0.8))


def _morirConElPadre():
    # pa pw-record: si el main se va (os._exit, kill) que no se quede
    # huerfano grabando pa siempre (PR_SET_PDEATHSIG = 1)
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, int(signal.SIGKILL))
    except Exception:
        pass


class OidoVoz:
    """mic -> texto con vosk, en su hilo; mete las frases a una cola.

    de donde sale el audio:
      - "pipewire": pw-record (la pi; remuestrea solo a 16 kHz y sigue
        al microfono por defecto, o sea la diadema usb aunque se conecte
        despues de arrancar)
      - "sounddevice": el VoiceRecognizer del equipo (laptop)

    modo "boton": solo escucha cuando se le pide (alternar(), lo llama
    el boton fisico) y se apaga solo tras autoApagarS sin oir nada.
    modo "continuo": siempre escuchando, como antes.

    mientras el sistema habla el microfono SIGUE abierto (antes se
    tiraba ese audio y por eso no entendia nada si le hablabas encima).
    pa que no se crea su propia voz: vosk da el tiempo de cada palabra,
    y las que cayeron mientras el sistema hablaba Y son palabras de lo
    que estaba diciendo se quitan (ecoFn). lo demas es del usuario."""

    FRECUENCIA = 16000
    # de las palabras oidas ENCIMA de la voz del sistema, si esta parte
    # son las que el sistema decia, el micro lo esta oyendo (eco)
    ECO_FRACCION = 0.5
    CONF_MIN_ENCIMADA = 0.5      # confianza de vosk pa creerle una encimada
    ECO_PA_MEDIO_DUPLEX = 6      # palabras de eco pa dejar de oir al hablar

    def __init__(self, modelPath, device=None, fuente="auto",
                 modo="continuo", autoApagarS=30.0):
        self.ok = False
        self.razon = ""
        self.modo = modo
        self.autoApagarS = autoApagarS
        self.alCambiar = None          # callable(escuchando): led/tonito
        self.alOir = None              # callable(texto): el usuario dijo algo
        # callable(t0, t1) -> textos que el sistema decia en ese rato
        self.ecoFn = None
        self._rec = None
        self._escuchando = modo != "boton"
        self._tVoz = time.time()
        self._sucio = False
        self._cola = None
        self._lock = threading.RLock()     # todo lo que toca a vosk
        self._sAudio = 0.0             # segundos de audio dados a vosk
        self._tramosEco = []           # (s0, s1, palabras del sistema)
        self._tParcial = 0.0           # ultima vez que iba a media frase
        self._tAbierto = 0.0
        self._oyoAlgo = False
        self.hayMic = fuente != "pipewire"   # pipewire: lo pone el hilo
        # si el micro SI oye a la bocina (sin diadema, o una que se
        # cuela mucho) se pasa solo a medio duplex: no oir mientras habla
        self.medioDuplex = False
        self._ecoOido = 0
        self._proc = None
        self._cerrado = False
        try:
            _vozypln()
            import shutil
            if not os.path.isdir(modelPath):
                raise FileNotFoundError(
                    f"No se encontró el modelo Vosk en '{modelPath}'")
            if fuente == "auto":
                fuente = ("pipewire" if shutil.which("pw-record")
                          and os.path.exists(os.path.join(
                              os.environ.get("XDG_RUNTIME_DIR", "/nope"),
                              "pipewire-0")) else "sounddevice")
            self.fuente = fuente
            if fuente == "pipewire":
                from vosk import Model, KaldiRecognizer, SetLogLevel
                SetLogLevel(-1)
                self._modelo = Model(modelPath)
                self._kaldi = KaldiRecognizer(self._modelo, self.FRECUENCIA)
                self._kaldi.SetWords(True)     # tiempos pa quitar el eco
            else:
                from voz_recog import VoiceRecognizer
                self._rec = VoiceRecognizer(modelPath, device=device)
            self.ok = True
        except FileNotFoundError as e:
            self.razon = str(e)
        except ImportError as e:
            self.razon = f"falta dependencia: {e}"
        except Exception as e:
            self.razon = str(e)

    @property
    def escuchando(self):
        return self._escuchando

    def escuchar(self, encendido):
        with self._lock:
            if encendido == self._escuchando:
                return encendido
            self._escuchando = encendido
            self._tVoz = time.time()
            if encendido:
                self._tAbierto = time.time()
                self._oyoAlgo = False
            if not encendido and self._sucio and self._cola is not None:
                # lo que alcanzo a decir antes de soltar el boton
                self._entregar(self._kaldi.FinalResult())
        if self.alCambiar is not None:
            try:
                self.alCambiar(encendido)
            except Exception:
                pass
        return encendido

    def alternar(self):
        return self.escuchar(not self._escuchando)

    def cerrar(self):
        """suelta el microfono (mata a pw-record)."""
        self._cerrado = True
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def usuarioHablando(self):
        """True si el micro esta abierto y el usuario va a media frase,
        o lo acaba de abrir con el boton y todavia no dice nada: la voz
        normal se espera pa no hablarle encima."""
        if not self._escuchando:
            return False
        ahora = time.time()
        if ahora - self._tParcial < 0.8:
            return True
        return (self.modo == "boton" and not self._oyoAlgo
                and ahora - self._tAbierto < 2.5)

    def _sinEco(self, r):
        """quita del resultado de vosk las palabras que son la voz del
        propio sistema colandose al microfono."""
        palabras = r.get("result")
        if not palabras or not self._tramosEco:
            return r.get("text", "").strip()
        marcadas = []                  # (palabra, encimada, esEco, conf)
        for w in palabras:
            encimada = esEco = False
            for s0, s1, dichas in self._tramosEco:
                if w["start"] < s1 and w["end"] > s0:
                    encimada = True
                    if _pareceEco(w["word"], dichas):
                        esEco = True
                        break
            marcadas.append((w["word"], encimada, esEco, w.get("conf", 1.0)))
        encimadas = sum(1 for m in marcadas if m[1])
        ecos = sum(1 for m in marcadas if m[2])
        if not encimadas or ecos < self.ECO_FRACCION * encimadas:
            # nada (o casi nada) de lo que decia el sistema se colo: lo
            # que se oyo encima de su voz es el usuario interrumpiendo.
            # una palabra suelta que coincide ("que hay enfrente" mientras
            # el sistema dice "al frente...") es casualidad y se queda;
            # lo que vosk oyo muy dudoso encima de la voz, no
            return " ".join(w for w, enc, _x, conf in marcadas
                            if not enc or conf >= self.CONF_MIN_ENCIMADA)
        # el micro SI esta oyendo al sistema: de esta frase solo vale lo
        # que cayo fuera de su voz (lo de adentro puede ser eco mal oido)
        self._ecoOido += ecos
        if self._ecoOido >= self.ECO_PA_MEDIO_DUPLEX and not self.medioDuplex:
            self.medioDuplex = True
            print("[VOZ] el micrófono oye la voz del sistema: desde ahora "
                  "no escucho mientras hablo", flush=True)
        return " ".join(w for w, enc, _x, _c in marcadas if not enc)

    def _entregar(self, resultadoJson):
        import json
        try:
            texto = self._sinEco(json.loads(resultadoJson))
        except (ValueError, KeyError, TypeError):
            texto = ""
        self._sucio = False
        self._tParcial = 0.0
        if texto and self._cola is not None:
            print(f"Usuario: {texto}", flush=True)
            self._tVoz = time.time()
            self._oyoAlgo = True
            if self.alOir is not None:
                try:
                    self.alOir(texto)    # antes de que el main conteste
                except Exception:
                    pass
            self._cola.put(texto)
        return texto

    def _anotarEco(self, s0, s1, tCaptura):
        if self.ecoFn is None:
            return None
        textos = self.ecoFn(tCaptura - (s1 - s0), tCaptura)
        if not textos:
            return None
        dichas = _palabrasEco(" ".join(textos))
        self._tramosEco.append((s0, s1, dichas))
        del self._tramosEco[:-400]       # ~50 s a 8 bloques por segundo
        return dichas

    def alimentar(self, datos, mudo=False, tCaptura=None):
        """un bloque de audio pcm16 mono 16 kHz (publico pa las pruebas:
        asi se le puede dar un wav sin microfono). tCaptura: cuando se
        grabo ese audio (time.time()), pa saber si el sistema hablaba.
        ojo: va con candado. el boton cierra el micro desde OTRO hilo
        (escuchar -> FinalResult) y vosk no aguanta dos hilos a la vez:
        tronaba todo el programa con un assert de kaldi (SIGABRT;
        reproducido, y es lo mas probable de la corrida de navegacion
        que termino con -6 el 2026-11-12)."""
        with self._lock:
            self._alimentar(datos, mudo, tCaptura)

    def _alimentar(self, datos, mudo, tCaptura):
        if not self._escuchando or mudo:
            if self._sucio:
                self._kaldi.Reset()      # no arrastrar audio a medias
                self._sucio = False
            return
        s0 = self._sAudio
        self._sAudio += len(datos) / (2.0 * self.FRECUENCIA)
        dichas = self._anotarEco(s0, self._sAudio,
                                 time.time() if tCaptura is None else tCaptura)
        if dichas and self.medioDuplex:
            if self._sucio:
                self._kaldi.Reset()
                self._sucio = False
            return
        self._sucio = True
        if self._kaldi.AcceptWaveform(datos):
            self._entregar(self._kaldi.Result())
            return
        import json
        try:
            parcial = json.loads(self._kaldi.PartialResult()).get("partial", "")
        except ValueError:
            parcial = ""
        if dichas:
            # mientras el sistema habla, su propia voz no cuenta
            parcial = " ".join(w for w in parcial.split()
                               if not _pareceEco(w, dichas))
            if len(parcial.split()) >= 2 and self.alOir is not None:
                # el usuario le esta hablando encima: que se calle ya,
                # sin esperar a que acabe la frase
                try:
                    self.alOir(parcial)
                except Exception:
                    pass
        if parcial:
            self._tParcial = time.time()
            self._tVoz = time.time()             # esta hablando
        elif (self.modo == "boton" and self.autoApagarS
              and time.time() - self._tVoz > self.autoApagarS):
            self.escuchar(False)

    def _loopPipewire(self, muteFn):
        import subprocess
        import fcntl
        import struct
        import termios
        import audio_pi
        bloque = 4000                            # 1/8 de segundo
        avisado = False
        while not self._cerrado:
            # sin microfono NO se graba: pipewire le daria a pw-record el
            # monitor de la bocina y el sistema se oiria a si mismo
            mic = audio_pi.microfono()
            self.hayMic = mic is not None
            if mic is None:
                if not avisado:
                    print("[VOZ] no hay micrófono: en cuanto se conecte la "
                          "diadema empiezo a escuchar", flush=True)
                    avisado = True
                time.sleep(3)
                continue
            if avisado:
                print(f"[VOZ] micrófono conectado: {mic}", flush=True)
                avisado = False
            try:
                proc = subprocess.Popen(
                    ["pw-record", "--target", mic,
                     "-P", "{ node.dont-reconnect = true }",
                     "--raw", "--rate", str(self.FRECUENCIA),
                     "--channels", "1", "--format", "s16",
                     "--latency", "60ms", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=0, preexec_fn=_morirConElPadre)
            except OSError as e:
                print(f"[VOZ] pw-record fallo: {e}")
                return
            self._proc = proc
            try:
                fd = proc.stdout.fileno()
                pend = b""
                tRevision = time.time()
                while True:
                    datos = proc.stdout.read(bloque)
                    if not datos:
                        break
                    if time.time() - tRevision > 5.0:
                        # ¿sigue ahi el microfono? (si lo desconectan,
                        # pipewire podria pasarnos al monitor de la bocina)
                        tRevision = time.time()
                        if audio_pi.microfono() != mic:
                            break
                    pend += datos
                    if len(pend) < bloque:
                        continue
                    n = len(pend) & ~1           # muestras completas
                    datos, pend = pend[:n], pend[n:]
                    # lo que sigue esperando en el tubo es audio VIEJO: se
                    # le resta pa saber cuando se grabo este bloque
                    try:
                        cola = struct.unpack("i", fcntl.ioctl(
                            fd, termios.FIONREAD, b"\0\0\0\0"))[0]
                    except OSError:
                        cola = 0
                    tCaptura = time.time() - cola / (2.0 * self.FRECUENCIA)
                    self.alimentar(datos, mudo=muteFn(), tCaptura=tCaptura)
            except Exception as e:
                print(f"[VOZ] mic fallo: {e}")
            finally:
                proc.kill()
                self.hayMic = False
            time.sleep(2)

    def _loopEquipo(self, muteFn):
        while True:
            if not self._escuchando:
                time.sleep(0.1)
                continue
            try:
                texto = self._rec.listen(timeout=5.0)
            except Exception as e:
                print(f"[VOZ] mic fallo: {e}")
                time.sleep(2)
                continue
            # si el tts estaba hablando, lo mas seguro es que el mic
            # se oyo a si mismo
            if texto and not muteFn():
                self._cola.put(texto)

    def start(self, colaTexto, muteFn=lambda: False, ecoFn=None):
        """muteFn(): True = tirar el audio (medio duplex, como antes).
        ecoFn(t0, t1): lo que el sistema decia en ese rato; con esto el
        micro puede seguir abierto mientras el sistema habla."""
        if not self.ok:
            return
        self._cola = colaTexto
        if ecoFn is not None:
            self.ecoFn = ecoFn
        loop = (self._loopPipewire if self.fuente == "pipewire"
                else self._loopEquipo)
        threading.Thread(target=loop, args=(muteFn,), daemon=True).start()


AYUDA_VOZ = ("Puede decir: llévame a un lugar, pausa, continúa, detener, "
             "otra ruta, dónde estoy, qué hay enfrente, o repite.")

def _decirDistancia(r):
    if r < 1.0:
        return f"{max(10, int(round(r * 10)) * 10)} centímetros"
    m = int(round(r))
    return "1 metro" if m == 1 else f"{m} metros"


def _contracciones(texto):
    # "llegamos a el punto" -> "al punto", "de el" -> "del"
    return texto.replace(" a el ", " al ").replace(" de el ", " del ")


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
# prender/apagar las alertas por voz (rf-06: antes solo con el boton)
_ALERTAS_OFF = re.compile(r"\b(apaga|apagar|desactiva|desactivar|quita|quitar|"
                          r"silencia|silenciar)\b.*\b(alertas?|avisos?|"
                          r"pitidos?)\b")
_ALERTAS_ON = re.compile(r"\b(prende|prender|enciende|encender|activa|"
                         r"activar|pon|poner)\b.*\b(alertas?|avisos?|"
                         r"pitidos?)\b")


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
        self.alAlertas = None     # callable(encender) que pone el main

    def _bonito(self, r):
        return self.dialogo.friendly_response(r) if self.amigable else None

    def manejarTexto(self, texto, navFrame=None):
        if not texto or not texto.strip():
            return []

        from PLN_oneimport import normalize_text
        limpio = normalize_text(texto)
        if limpio in _SALIR:
            return [NavMessage(self.dialogo.goodbye(), priority=2,
                               kind="despedida")]
        if self.alAlertas is not None:
            apagar = _ALERTAS_OFF.search(limpio)
            if apagar or _ALERTAS_ON.search(limpio):
                self.alAlertas(not apagar)
                return [NavMessage("Alertas desactivadas." if apagar
                                   else "Alertas activadas.", priority=1,
                                   kind="respuesta")]

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
            # ojo: el sistema NO avisa a nadie (no hay contactos de
            # emergencia); solo se detiene y lo dice tal cual
            aviso = ("Tranquilo, estoy contigo. Detuve la navegación. "
                     "Quédate donde estás y pide ayuda a alguien cercano.")
            msgs = [NavMessage(aviso, priority=0, kind="alerta")]
            if self.navigator.status.session == "activo":
                msgs += self.navigator.handle_command(
                    NavCommand("pausar"), navFrame)
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

    def ciclo(self, navFrame, mgr, alertas=None, recalcular=False):
        st0 = self.navigator.status
        label, dicho = st0.goal_label, st0.goal_nombre
        msgs = self.navigator.update(navFrame, mgr, alertas,
                                     replan=recalcular)
        for m in msgs:
            if self.amigable:
                if m.kind == "instruccion" and m.action is not None:
                    amigo = self._instruccionAmigable(m.action, navFrame)
                    if amigo:
                        m.text = amigo
                elif m.text == "Ha llegado a su destino.":
                    m.text = _contracciones(self.dialogo.arrival_message(
                        self._nombreDestino(label, dicho)))
            if m.kind == "instruccion":
                self.ultimaInstruccion = m.text
        return msgs

    def _nombreDestino(self, label, dicho):
        # se repite lo que dijo el usuario ("la salida"); si no hay
        # (destino por punto o por landmark), el nombre de la clase
        from nombres import nombreEs, conArticulo
        if dicho:
            return self.dialogo.DISPLAY_NAMES.get(dicho) or conArticulo(dicho)
        return nombreEs(label)

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
                if d.label == "exit sign":
                    continue          # afuera es cualquier letrero verde
                if lo <= d.bearing <= hi and d.range_est is not None:
                    if cerca is None or d.range_est < cerca.range_est:
                        cerca = d
            if cerca is not None:
                extra = " en movimiento" if cerca.moving else ""
                from nombres import nombreEs
                tag = nombreEs(cerca.label, "indef")
                if cerca.hazard:
                    tag += " en el piso"
                if cerca.label == "door" and cerca.estado:
                    tag += f" {cerca.estado}"
                partes.append(f"{nombre}, {tag}{extra} a "
                              f"{_decirDistancia(cerca.range_est)}")
            else:
                d_libre = {"Al frente": frame.d_front,
                           "A la izquierda": frame.d_left,
                           "A la derecha": frame.d_right}[nombre]
                if d_libre >= 3.99:
                    partes.append(f"{nombre} libre")
                else:
                    partes.append(f"{nombre}, obstáculo a "
                                  f"{_decirDistancia(d_libre)}")
        if frame.room:
            partes.append(f"Está en {frame.room}")
        return ". ".join(partes) + "."


def armarInteraccion(navigator, usarVoz=False, usarTts=False,
                     voskModel=None, micDevice=None, echo=True,
                     motorVoz="auto", modoEscucha="continuo"):
    """fabrica todo el stack de interaccion con fallbacks: sin modelo
    vosk o sin mic avisa y sigue solo con teclado."""
    voz = ColaVoz(hablar=usarTts, echo=echo, motor=motorVoz)
    inter = Interaccion(navigator)
    oido = None
    if usarVoz:
        oido = OidoVoz(voskModel or "models/vosk-model-small-es-0.42",
                       device=micDevice, modo=modoEscucha)
        if oido.ok:
            print("[VOZ] reconocimiento de voz (vosk) listo")
        else:
            print(f"[VOZ] reconocimiento no disponible: {oido.razon}")
            print("[VOZ] seguimos con comandos por teclado")
            oido = None
    return inter, voz, oido


def conectarOido(oido, voz, colaTexto, mudoAlHablar=False):
    """arranca el oido ya amarrado a la voz. en la pi (pipewire, con la
    diadema) el micro sigue abierto mientras el sistema habla y su
    propia voz se quita por texto; con el reconocedor del equipo
    (laptop con bocinas) se queda como antes: no oye mientras habla."""
    if oido is None:
        return
    # si el usuario va hablando la voz normal se espera, y en cuanto se
    # le entiende algo se corta lo que se estuviera diciendo
    voz.esperarUsuarioSi = oido.usuarioHablando
    oido.alOir = lambda _texto: voz.callar()
    if mudoAlHablar or getattr(oido, "fuente", "") != "pipewire":
        oido.start(colaTexto, muteFn=lambda: voz.hablando)
    else:
        oido.start(colaTexto, ecoFn=voz.dichoEntre)
